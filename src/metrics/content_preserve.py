import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ---------------------------------------------------------------------------
# LDC utilities (from ldc.py)
# ---------------------------------------------------------------------------


def image_normalization(img, img_min=0, img_max=255, epsilon=1e-12):
    img = np.float32(img)
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img


def postprocess_edges(tensor, img_shape=None, arg=None, is_inchannel=False):
    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)

    image_shape = [[256], [256]]
    image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

    idx = 0
    for i_shape in image_shape:
        tmp = tensor[:, idx, ...]
        tmp = np.squeeze(tmp)

        preds = []
        fuse_num = tmp.shape[0] - 1
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)
            preds.append(tmp_img)

        average = np.array(preds, dtype=np.float32)
        average = np.uint8(np.mean(average, axis=0))

        idx += 1
    return 255 - average


def weight_init(m):
    if isinstance(m, (nn.Conv2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)
        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, mean=0.0)

        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

    # for fusion layer
    if isinstance(m, (nn.ConvTranspose2d,)):
        torch.nn.init.xavier_normal_(m.weight, gain=1.0)

        if m.weight.data.shape[1] == torch.Size([1]):
            torch.nn.init.normal_(m.weight, std=0.1)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class CoFusion(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CoFusion, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, out_ch, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.norm_layer1 = nn.GroupNorm(4, 32)

    def forward(self, x):
        attn = self.relu(self.norm_layer1(self.conv1(x)))
        attn = F.softmax(self.conv3(attn), dim=1)
        return ((x * attn).sum(1)).unsqueeze(1)


class _DenseLayer(nn.Sequential):
    def __init__(self, input_features, out_features):
        super(_DenseLayer, self).__init__()

        self.add_module('conv1', nn.Conv2d(input_features, out_features,
                                           kernel_size=3, stride=1, padding=2, bias=True)),
        self.add_module('norm1', nn.BatchNorm2d(out_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(out_features, out_features,
                                           kernel_size=3, stride=1, bias=True)),
        self.add_module('norm2', nn.BatchNorm2d(out_features))

    def forward(self, x):
        x1, x2 = x
        new_features = super(_DenseLayer, self).forward(F.relu(x1))
        return 0.5 * (new_features + x2), x2


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, input_features, out_features):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(input_features, out_features)
            self.add_module('denselayer%d' % (i + 1), layer)
            input_features = out_features


class UpConvBlock(nn.Module):
    def __init__(self, in_features, up_scale):
        super(UpConvBlock, self).__init__()
        self.up_factor = 2
        self.constant_features = 16

        layers = self.make_deconv_layers(in_features, up_scale)
        assert layers is not None, layers
        self.features = nn.Sequential(*layers)

    def make_deconv_layers(self, in_features, up_scale):
        layers = []
        all_pads = [0, 0, 1, 3, 7]
        for i in range(up_scale):
            kernel_size = 2 ** up_scale
            pad = all_pads[up_scale]
            out_features = self.compute_out_features(i, up_scale)
            layers.append(nn.Conv2d(in_features, out_features, 1))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.ConvTranspose2d(
                out_features, out_features, kernel_size, stride=2, padding=pad))
            in_features = out_features
        return layers

    def compute_out_features(self, idx, up_scale):
        return 1 if idx == up_scale - 1 else self.constant_features

    def forward(self, x):
        return self.features(x)


class SingleConvBlock(nn.Module):
    def __init__(self, in_features, out_features, stride, use_bs=True):
        super(SingleConvBlock, self).__init__()
        self.use_bn = use_bs
        self.conv = nn.Conv2d(in_features, out_features, 1, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(self, in_features, mid_features, out_features=None,
                 stride=1, use_act=True):
        super(DoubleConvBlock, self).__init__()

        self.use_act = use_act
        if out_features is None:
            out_features = mid_features
        self.conv1 = nn.Conv2d(in_features, mid_features, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(mid_features)
        self.conv2 = nn.Conv2d(mid_features, out_features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.use_act:
            x = self.relu(x)
        return x


class LDC(nn.Module):
    """Definition of the LDC (DXtrem) edge-detection network."""

    def __init__(self):
        super(LDC, self).__init__()
        self.block_1 = DoubleConvBlock(3, 16, 16, stride=2)
        self.block_2 = DoubleConvBlock(16, 32, use_act=False)
        self.dblock_3 = _DenseBlock(2, 32, 64)
        self.dblock_4 = _DenseBlock(3, 64, 96)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # left skip connections
        self.side_1 = SingleConvBlock(16, 32, 2)
        self.side_2 = SingleConvBlock(32, 64, 2)

        # right skip connections
        self.pre_dense_2 = SingleConvBlock(32, 64, 2)
        self.pre_dense_3 = SingleConvBlock(32, 64, 1)
        self.pre_dense_4 = SingleConvBlock(64, 96, 1)

        # USNet
        self.up_block_1 = UpConvBlock(16, 1)
        self.up_block_2 = UpConvBlock(32, 1)
        self.up_block_3 = UpConvBlock(64, 2)
        self.up_block_4 = UpConvBlock(96, 3)
        self.block_cat = CoFusion(4, 4)

        self.apply(weight_init)

    def slice(self, tensor, slice_shape):
        t_shape = tensor.shape
        height, width = slice_shape
        if t_shape[-1] != slice_shape[-1]:
            new_tensor = F.interpolate(
                tensor, size=(height, width), mode='bicubic', align_corners=False)
        else:
            new_tensor = tensor
        return new_tensor

    def forward(self, x):
        assert x.ndim == 4, x.shape

        # Block 1
        block_1 = self.block_1(x)
        block_1_side = self.side_1(block_1)

        # Block 2
        block_2 = self.block_2(block_1)
        block_2_down = self.maxpool(block_2)
        block_2_add = block_2_down + block_1_side
        block_2_side = self.side_2(block_2_add)

        # Block 3
        block_3_pre_dense = self.pre_dense_3(block_2_down)
        block_3, _ = self.dblock_3([block_2_add, block_3_pre_dense])
        block_3_down = self.maxpool(block_3)
        block_3_add = block_3_down + block_2_side

        # Block 4
        block_2_resize_half = self.pre_dense_2(block_2_down)
        block_4_pre_dense = self.pre_dense_4(block_3_down + block_2_resize_half)
        block_4, _ = self.dblock_4([block_3_add, block_4_pre_dense])

        # upsampling blocks
        out_1 = self.up_block_1(block_1)
        out_2 = self.up_block_2(block_2)
        out_3 = self.up_block_3(block_3)
        out_4 = self.up_block_4(block_4)
        results = [out_1, out_2, out_3, out_4]

        # concatenate multiscale outputs
        block_cat = torch.cat(results, dim=1)
        block_cat = self.block_cat(block_cat)

        results.append(block_cat)
        return results


# ---------------------------------------------------------------------------
# Content-similarity computation (from calc_content_similiary.py)
# ---------------------------------------------------------------------------


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Weight file lives under ICTone/ckpt/metric_weights (a symlink to the shared
# eval-metric weight folder). Override with the ``ICTONE_METRICS_WEIGHTS`` env
# var if needed.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ICTONE_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_WEIGHTS_ROOT = os.environ.get(
    'ICTONE_METRICS_WEIGHTS',
    os.path.join(_ICTONE_ROOT, 'ckpt', 'metric_weights'),
)
DEFAULT_LDC_WEIGHTS = os.path.join(_WEIGHTS_ROOT, 'LDC', 'ldc.pth')


def build_ldc_model(weights_path):
    model = LDC()
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    return model


def calculate_ldc_edge(image_path, ldc_model):
    image = Image.open(image_path).convert('RGB')

    # NOTE: PIL's `Image.size` returns (width, height); the original code uses
    # the names `h, w` in that order. We preserve the original variable order
    # so that the resizing behavior matches the reference implementation.
    h, w = image.size
    h = int(h - h % 32)
    w = int(w - w % 32)

    mean = torch.tensor([103.939, 116.779, 123.68]).to(DEVICE)
    mean = mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

    image = transforms.functional.resize(image, (w, h))
    image = transforms.functional.to_tensor(image)[None, ...].to(DEVICE) * 255

    with torch.no_grad():
        edges = ldc_model(image - mean)
    avg_edge = postprocess_edges(edges)
    avg_edge = torch.from_numpy(avg_edge).unsqueeze(0).unsqueeze(0) / 255

    return avg_edge


def _ssim_of_edges(ldc_model, path_a, path_b):
    edge_a = calculate_ldc_edge(path_a, ldc_model)
    edge_b = calculate_ldc_edge(path_b, ldc_model)
    edge_a = edge_a[0].permute(1, 2, 0).cpu().numpy()
    edge_b = edge_b[0].permute(1, 2, 0).cpu().numpy()
    # Two source images may have different sizes (e.g. gt vs reference), which
    # yields edge maps of different spatial dims. Align edge_b to edge_a.
    if edge_a.shape != edge_b.shape:
        edge_b = cv2.resize(
            edge_b, (edge_a.shape[1], edge_a.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        if edge_b.ndim == 2:
            edge_b = edge_b[..., None]
    return float(structural_similarity(
        edge_a, edge_b, channel_axis=-1, data_range=1.0,
    ))


def _read_list(list_path):
    """Return list of dicts with keys content/reference/gt/pred."""
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


VALID_KEYS = ('content', 'reference', 'gt', 'pred')


def _parse_pair(pair_str):
    try:
        a, b = pair_str.split(':')
    except ValueError:
        raise ValueError(f'--pair must be like "pred:gt", got {pair_str!r}')
    if a not in VALID_KEYS or b not in VALID_KEYS:
        raise ValueError(f'--pair keys must be in {VALID_KEYS}, got {pair_str!r}')
    if a == b:
        raise ValueError(f'--pair keys must differ, got {pair_str!r}')
    return a, b


def _run_list_mode(args):
    src_key, tgt_key = _parse_pair(args.pair)
    rows = _read_list(args.list)
    print(f'read {len(rows)} rows from {args.list}  (pair: {src_key} vs {tgt_key})')

    ldc_model = build_ldc_model(args.ldc_weights)

    scores = []
    pbar = tqdm(rows, total=len(rows), unit='pair')
    for row in pbar:
        s = _ssim_of_edges(ldc_model, row[src_key], row[tgt_key])
        scores.append((row[src_key], s))
        pbar.set_description(
            'Avg SSIM(edge, {0} vs {1}): {2:.4f}'.format(
                src_key, tgt_key,
                float(np.mean([v for _, v in scores]))
            )
        )

    mean = float(np.mean([v for _, v in scores])) if scores else float('nan')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(f'# metric: content_preserve (LDC-edge SSIM, {src_key} vs {tgt_key})\n')
        f.write(f'# list: {args.list}\n')
        f.write(f'# pairs: {len(scores)}\n')
        for src_path, s in scores:
            f.write(f'{src_path}\t{s:.6f}\n')
        f.write(f'Avg\t{mean:.6f}\n')
    print(f'wrote {args.out}')
    print(f'Final Avg Content Preserve Score ({src_key} vs {tgt_key}): {mean:.4f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--list', type=str, default=None,
                        help='4-column txt list: content reference gt pred.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output txt with per-pair scores + average.')
    parser.add_argument('--pair', type=str, default='pred:gt',
                        help='Which two columns to compare, e.g. "pred:gt" (default), '
                             '"gt:reference". Keys in {content, reference, gt, pred}.')
    parser.add_argument('--result-folder', type=str, default=None)
    parser.add_argument('--content-folder', type=str, default=None)
    parser.add_argument('--ldc-weights', type=str, default=DEFAULT_LDC_WEIGHTS,
                        help='Path to ldc.pth (defaults to the NeuralPreset copy).')
    args = parser.parse_args()

    if not os.path.exists(args.ldc_weights):
        print('Cannot find the ldc weights: {0}'.format(args.ldc_weights))
        return

    if args.list is not None:
        if args.out is None:
            parser.error('--out is required with --list.')
        if not os.path.exists(args.list):
            print('Cannot find the list file: {0}'.format(args.list))
            return
        _run_list_mode(args)
        return

    if not args.result_folder or not args.content_folder:
        parser.error('Provide either --list, or both --result-folder and --content-folder.')

    if not os.path.exists(args.result_folder):
        print('Cannot find the result folder: {0}'.format(args.result_folder))
        return
    if not os.path.exists(args.content_folder):
        print('Cannot find the content folder: {0}'.format(args.content_folder))
        return

    ldc_model = build_ldc_model(args.ldc_weights)

    result_files = sorted(os.listdir(args.result_folder))
    content_files = sorted(os.listdir(args.content_folder))
    assert len(result_files) == len(content_files), (
        'result/content folders must have the same number of files: '
        f'{len(result_files)} vs {len(content_files)}'
    )

    print('-' * 80)
    print('result folder:  {0}'.format(args.result_folder))
    print('content folder: {0}'.format(args.content_folder))
    print('ldc weights:    {0}'.format(args.ldc_weights))
    print('device:         {0}'.format(DEVICE))
    print('-' * 80)

    content_similiary_scores = []
    pbar = tqdm(result_files, total=len(result_files), unit='file')
    for fname in pbar:
        result_path = os.path.join(args.result_folder, fname)
        content_path = os.path.join(args.content_folder, fname)
        score = _ssim_of_edges(ldc_model, result_path, content_path)
        content_similiary_scores.append(score)
        pbar.set_description(
            'Avg Content Similiary Score: {0:.4f}'.format(
                float(np.mean(np.asarray(content_similiary_scores)))
            )
        )

    print('-' * 80)
    print('Final Avg Content Similiary Score: {0:.4f}'.format(
        float(np.mean(np.asarray(content_similiary_scores)))
    ))


if __name__ == '__main__':
    main()
