import argparse
import os
from math import log, pi

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg as la
from torchvision import transforms


# ---------------------------------------------------------------------------
# flow.py contents
# ---------------------------------------------------------------------------


def logabs(x):
    return torch.log(torch.abs(x))


class ActNorm(nn.Module):
    def __init__(self, in_channel, logdet=True):
        super().__init__()
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (flatten.mean(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                    .permute(1, 0, 2, 3))
            std = (flatten.std(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
                   .permute(1, 0, 2, 3))
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        _, _, height, width = input.shape
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)
        logdet = height * width * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class InvConv2d(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.linalg.qr(weight)
        weight = q.unsqueeze(2).unsqueeze(3)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, _, height, width = input.shape
        out = F.conv2d(input, self.weight)
        logdet = (
            height * width * torch.slogdet(self.weight.squeeze().double())[1].float()
        )
        return out, logdet

    def reverse(self, output):
        return F.conv2d(
            output, self.weight.squeeze().inverse().unsqueeze(2).unsqueeze(3)
        )


class InvConv2dLU(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        weight = np.random.randn(in_channel, in_channel)
        q, _ = la.qr(weight)
        w_p, w_l, w_u = la.lu(q.astype(np.float32))
        w_s = np.diag(w_u)
        w_u = np.triu(w_u, 1)
        u_mask = np.triu(np.ones_like(w_u), 1)
        l_mask = u_mask.T

        w_p = torch.from_numpy(w_p)
        w_l = torch.from_numpy(w_l)
        w_s = torch.from_numpy(w_s)
        w_u = torch.from_numpy(w_u)

        self.register_buffer("w_p", w_p)
        self.register_buffer("u_mask", torch.from_numpy(u_mask))
        self.register_buffer("l_mask", torch.from_numpy(l_mask))
        self.register_buffer("s_sign", torch.sign(w_s))
        self.register_buffer("l_eye", torch.eye(l_mask.shape[0]))
        self.w_l = nn.Parameter(w_l)
        self.w_s = nn.Parameter(logabs(w_s))
        self.w_u = nn.Parameter(w_u)

    def forward(self, input):
        _, _, height, width = input.shape
        weight = self.calc_weight()
        out = F.conv2d(input, weight)
        logdet = height * width * torch.sum(self.w_s)
        return out, logdet

    def calc_weight(self):
        weight = (
            self.w_p
            @ (self.w_l * self.l_mask + self.l_eye)
            @ ((self.w_u * self.u_mask) + torch.diag(self.s_sign * torch.exp(self.w_s)))
        )
        return weight.unsqueeze(2).unsqueeze(3)

    def reverse(self, output):
        weight = self.calc_weight()
        return F.conv2d(output, weight.squeeze().inverse().unsqueeze(2).unsqueeze(3))


class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)
        return out


class AffineCoupling(nn.Module):
    def __init__(self, in_channel, filter_size=512, affine=True):
        super().__init__()
        self.affine = affine
        self.net = nn.Sequential(
            nn.Conv2d(in_channel // 2, filter_size, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(filter_size, filter_size, 1),
            nn.ReLU(inplace=True),
            ZeroConv2d(filter_size, in_channel if self.affine else in_channel // 2),
        )
        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            out_b = (in_b + t) * s
            logdet = torch.sum(torch.log(s).view(input.shape[0], -1), 1)
        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None
        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)
        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            s = torch.sigmoid(log_s + 2)
            in_b = out_b / s - t
        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out
        return torch.cat([out_a, in_b], 1)


class Flow(nn.Module):
    def __init__(self, in_channel, affine=True, conv_lu=True):
        super().__init__()
        self.actnorm = ActNorm(in_channel)
        if conv_lu:
            self.invconv = InvConv2dLU(in_channel)
        else:
            self.invconv = InvConv2d(in_channel)
        self.coupling = AffineCoupling(in_channel, affine=affine)

    def forward(self, input):
        out, logdet = self.actnorm(input)
        out, det1 = self.invconv(out)
        out, det2 = self.coupling(out)
        logdet = logdet + det1
        if det2 is not None:
            logdet = logdet + det2
        return out, logdet

    def reverse(self, output):
        input = self.coupling.reverse(output)
        input = self.invconv.reverse(input)
        input = self.actnorm.reverse(input)
        return input


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()
        squeeze_dim = in_channel * 4
        self.flows = nn.ModuleList()
        for _ in range(n_flow):
            self.flows.append(Flow(squeeze_dim, affine=affine, conv_lu=conv_lu))
        self.split = split
        if split:
            self.prior = ZeroConv2d(in_channel * 2, in_channel * 4)
        else:
            self.prior = ZeroConv2d(in_channel * 4, in_channel * 8)

    def forward(self, input):
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)

        logdet = 0
        for flow in self.flows:
            out, det = flow(out)
            logdet = logdet + det

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
        else:
            one = torch.ones_like(out)
            mean, log_sd = self.prior(one).chunk(2, 1)
            log_p = gaussian_log_p(out, mean, log_sd)
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None, reconstruct=False):
        input = output
        if reconstruct:
            if self.split:
                input = torch.cat([output, eps], 1)
            else:
                input = eps
        else:
            if self.split:
                mean, log_sd = self.prior(input).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = torch.cat([output, z], 1)
            else:
                one = torch.ones_like(input)
                mean, log_sd = self.prior(one).chunk(2, 1)
                z = gaussian_sample(eps, mean, log_sd)
                input = z

        for flow in self.flows[::-1]:
            input = flow.reverse(input)

        b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )
        return unsqueezed


class Glow(nn.Module):
    def __init__(self, in_channel, n_flow, n_block, affine=True, conv_lu=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for _ in range(n_block - 1):
            self.blocks.append(Block(n_channel, n_flow, affine=affine, conv_lu=conv_lu))
            n_channel *= 2
        self.blocks.append(Block(n_channel, n_flow, split=False, affine=affine))

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []
        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det
            if log_p is not None:
                log_p_sum = log_p_sum + log_p
        return log_p_sum, logdet, z_outs

    def reverse(self, z_list, reconstruct=True, cd_map=False):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1], reconstruct=reconstruct)
            else:
                input = block.reverse(input, z_list[-(i + 1)], reconstruct=reconstruct)
        return input


# ---------------------------------------------------------------------------
# model.py contents
# ---------------------------------------------------------------------------


class CDFlow(nn.Module):
    def __init__(self):
        super(CDFlow, self).__init__()
        self.glow = Glow(3, 8, 6, affine=True, conv_lu=True)

    def coordinate_transform(self, x_hat, rev=False):
        if not rev:
            log_p, logdet, x_hat = self.glow(x_hat)
            return log_p, logdet, x_hat
        else:
            x_hat = self.glow.reverse(x_hat)
            return x_hat

    def forward(self, x, y):
        log_p_x, logdet_x, x_hat = self.coordinate_transform(x, rev=False)
        log_p_y, logdet_y, y_hat = self.coordinate_transform(y, rev=False)

        b = x_hat[0].shape[0]
        x_hat_1, y_hat_1 = x_hat[0].view(b, -1), y_hat[0].view(b, -1)
        x_hat_2, y_hat_2 = x_hat[1].view(b, -1), y_hat[1].view(b, -1)
        x_hat_3, y_hat_3 = x_hat[2].view(b, -1), y_hat[2].view(b, -1)
        x_hat_4, y_hat_4 = x_hat[3].view(b, -1), y_hat[3].view(b, -1)
        x_hat_5, y_hat_5 = x_hat[4].view(b, -1), y_hat[4].view(b, -1)
        x_hat_6, y_hat_6 = x_hat[5].view(b, -1), y_hat[5].view(b, -1)

        x_cat_65 = torch.cat((x_hat_6, x_hat_5), dim=1)
        y_cat_65 = torch.cat((y_hat_6, y_hat_5), dim=1)
        x_cat_654 = torch.cat((x_hat_6, x_hat_5, x_hat_4), dim=1)
        y_cat_654 = torch.cat((y_hat_6, y_hat_5, y_hat_4), dim=1)
        x_cat_6543 = torch.cat((x_hat_6, x_hat_5, x_hat_4, x_hat_3), dim=1)
        y_cat_6543 = torch.cat((y_hat_6, y_hat_5, y_hat_4, y_hat_3), dim=1)
        x_cat_65432 = torch.cat((x_hat_6, x_hat_5, x_hat_4, x_hat_3, x_hat_2), dim=1)
        y_cat_65432 = torch.cat((y_hat_6, y_hat_5, y_hat_4, y_hat_3, y_hat_2), dim=1)
        x_cat_654321 = torch.cat((x_hat_6, x_hat_5, x_hat_4, x_hat_3, x_hat_2, x_hat_1), dim=1)
        y_cat_654321 = torch.cat((y_hat_6, y_hat_5, y_hat_4, y_hat_3, y_hat_2, y_hat_1), dim=1)

        def _rmse(x_cat, y_cat):
            diff = (x_cat - y_cat).view(x_cat.shape[0], -1).unsqueeze(1)
            out = torch.sqrt(1e-8 + torch.matmul(diff, diff.transpose(-2, -1)) / diff.shape[2])
            return out.squeeze(2)

        mse6 = _rmse(x_hat_6, y_hat_6)
        mse65 = _rmse(x_cat_65, y_cat_65)
        mse654 = _rmse(x_cat_654, y_cat_654)
        mse6543 = _rmse(x_cat_6543, y_cat_6543)
        mse65432 = _rmse(x_cat_65432, y_cat_65432)
        mse654321 = _rmse(x_cat_654321, y_cat_654321)

        return mse654321, mse65432, mse6543, mse654, mse65, mse6, log_p_x, logdet_x, log_p_y, logdet_y


# ---------------------------------------------------------------------------
# Preprocessing / inference helpers
# ---------------------------------------------------------------------------


# Matches the `test=True` transform in CD-Flow/DataLoader.py:CD_128.
_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize(1024),
    transforms.CenterCrop(1024),
    transforms.ToTensor(),
])

# Weight file lives under ICTone/ckpt/metric_weights (a symlink to the shared
# eval-metric weight folder). Override with the ``ICTONE_METRICS_WEIGHTS`` env
# var if needed.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ICTONE_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_WEIGHTS_ROOT = os.environ.get(
    'ICTONE_METRICS_WEIGHTS',
    os.path.join(_ICTONE_ROOT, 'ckpt', 'metric_weights'),
)
DEFAULT_WEIGHTS = os.path.join(_WEIGHTS_ROOT, 'CDFlow', 'ModelParams_Best_val.pt')


def load_cdflow(weights_path, device):
    net = CDFlow().to(device)
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    # The checkpoint was saved from nn.DataParallel — strip the `module.` prefix.
    state_dict = {
        (k[len('module.'):] if k.startswith('module.') else k): v
        for k, v in state_dict.items()
    }
    # ActNorm layers guard on the `initialized` buffer; the checkpoint already
    # has it set to 1, so `initialize` will NOT run during forward and thus the
    # very first forward pass reproduces the original test-time behavior.
    net.load_state_dict(state_dict)
    net.eval()
    return net


def _load_image(path, device):
    img = Image.open(path).convert('RGB')
    tensor = _TEST_TRANSFORM(img).unsqueeze(0).to(device)
    return tensor


def color_difference(image1_path, image2_path, weights_path=DEFAULT_WEIGHTS, device=None):
    """Compute the color-difference score between two image files.

    Returns a Python float — the `mse654321` output of CDFlow (same as
    `score` in the reference `test.py`).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    net = load_cdflow(weights_path, device)
    x = _load_image(image1_path, device)
    y = _load_image(image2_path, device)

    with torch.no_grad():
        score, *_ = net(x, y)

    return float(score.squeeze().cpu().item())


def _read_list(list_path):
    rows = []
    list_dir = os.path.dirname(os.path.abspath(list_path))
    with open(list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            paths = [
                path if os.path.isabs(path)
                else os.path.normpath(os.path.join(list_dir, path))
                for path in parts[:4]
            ]
            rows.append(dict(zip(('content', 'reference', 'gt', 'pred'), paths)))
    return rows


def _score_pair(net, path_a, path_b, device):
    x = _load_image(path_a, device)
    y = _load_image(path_b, device)
    with torch.no_grad():
        score, *_ = net(x, y)
    return float(score.squeeze().cpu().item())


class _PredGtDataset(torch.utils.data.Dataset):
    """Loads (pred, gt) image pairs with the same transform as `_load_image`.

    Returned tensors are on CPU; the training loop moves them to `device`
    with `non_blocking=True` after `pin_memory` in the DataLoader.
    """

    def __init__(self, rows, transform=_TEST_TRANSFORM):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        pred_img = Image.open(row['pred']).convert('RGB')
        gt_img = Image.open(row['gt']).convert('RGB')
        pred_t = self.transform(pred_img)
        gt_t = self.transform(gt_img)
        return pred_t, gt_t, row['pred']


def _run_list_mode(list_path, out_path, weights_path, device,
                   batch_size=8, num_workers=4):
    rows = _read_list(list_path)
    print(f'read {len(rows)} rows from {list_path}')

    net = load_cdflow(weights_path, device)

    dataset = _PredGtDataset(rows)
    pin = (device.type == 'cuda')
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        drop_last=False,
    )

    from tqdm import tqdm  # local import to avoid extra top-level dep
    scores = []
    running_sum = 0.0
    pbar = tqdm(loader, total=len(loader), unit='batch')
    for xb, yb, paths in pbar:
        xb = xb.to(device, non_blocking=pin)
        yb = yb.to(device, non_blocking=pin)
        with torch.no_grad():
            score, *_ = net(xb, yb)
        score = score.detach().float().view(-1).cpu().tolist()
        for p, s in zip(paths, score):
            scores.append((p, s))
            running_sum += s
        pbar.set_description(
            'Avg CDFlow(pred, gt): {0:.4f}'.format(running_sum / len(scores))
        )

    mean = (running_sum / len(scores)) if scores else float('nan')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write('# metric: color_difference (CDFlow mse654321 between pred and gt)\n')
        f.write(f'# list: {list_path}\n')
        f.write(f'# pairs: {len(scores)}\n')
        for pred_path, s in scores:
            f.write(f'{pred_path} {s:.6f}\n')
        f.write(f'AVERAGE {mean:.6f}\n')
    print(f'wrote {out_path}')
    print(f'Final Avg Color Difference: {mean:.4f}')


def main():
    parser = argparse.ArgumentParser(description='CDFlow color-difference.')
    parser.add_argument('--image1', type=str, default=None)
    parser.add_argument('--image2', type=str, default=None)
    parser.add_argument('--list', type=str, default=None,
                        help='4-column txt list: content reference gt pred.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output txt with per-pair scores + average.')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS,
                        help='Path to ModelParams_Best_val.pt.')
    parser.add_argument('--device', type=str, default=None,
                        help='"cuda" or "cpu" (defaults to cuda if available).')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for list mode (default: 8).')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader worker count for list mode (default: 4).')
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else \
        torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.weights):
        raise FileNotFoundError(args.weights)

    if args.list is not None:
        if args.out is None:
            parser.error('--out is required with --list.')
        if not os.path.exists(args.list):
            raise FileNotFoundError(args.list)
        _run_list_mode(args.list, args.out, args.weights, device,
                       batch_size=args.batch_size,
                       num_workers=args.num_workers)
        return

    if not (args.image1 and args.image2):
        parser.error('Provide either --list, or both --image1 and --image2.')
    for p in (args.image1, args.image2):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    score = color_difference(args.image1, args.image2, args.weights, str(device))
    print('Color Difference Score: {0:.6f}'.format(score))


if __name__ == '__main__':
    main()
