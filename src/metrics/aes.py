from __future__ import annotations

import argparse
import csv
import gzip
import html
import math
import os
import sys
from collections import OrderedDict
from functools import lru_cache
from typing import List, Union

import ftfy
import regex as re
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from tqdm import tqdm

try:
    from torchvision.transforms import InterpolationMode
    _BICUBIC = InterpolationMode.BICUBIC
except ImportError:  # pragma: no cover — very old torchvision
    _BICUBIC = Image.BICUBIC


# ---------------------------------------------------------------------------
# Weight file locations
# ---------------------------------------------------------------------------
# All model weights (both the CLIP backbone in OpenAI .pt format and the
# AesCLIP fine-tunes) live under ICTone/ckpt/metric_weights (a symlink to the
# shared eval-metric weight folder). Override the location with the
# ``ICTONE_METRICS_WEIGHTS`` env var if needed.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ICTONE_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
_CKPT_DIR = os.path.join(_ICTONE_ROOT, 'ckpt')
_WEIGHTS_ROOT = os.environ.get(
    'ICTONE_METRICS_WEIGHTS',
    os.path.join(_CKPT_DIR, 'metric_weights'),
)
CLIP_BACKBONE = os.path.join(_WEIGHTS_ROOT, 'clip-vit-base-patch16', 'ViT-B-16-openai.pt')
AESCLIP_WEIGHT = os.path.join(_WEIGHTS_ROOT, 'AesCLIP', 'AesCLIP')
IAA_WEIGHT = os.path.join(_WEIGHTS_ROOT, 'AesCLIP', 'AesCLIP_IAA_tuned')
IMG_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tif', '.tiff')

# BPE vocab (ships alongside this file).
_BPE_PATH = os.path.join(_HERE, 'bpe_simple_vocab_16e6.txt.gz')


# =============================================================================
# Inlined CLIP tokenizer (from OpenAI CLIP simple_tokenizer.py)
# =============================================================================


@lru_cache()
def _bytes_to_unicode():
    """Reversible byte↔unicode map for BPE (see OpenAI GPT-2 tokenizer)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def _get_pairs(word):
    pairs = set()
    prev = word[0]
    for ch in word[1:]:
        pairs.add((prev, ch))
        prev = ch
    return pairs


def _basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def _whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


class _SimpleTokenizer:
    def __init__(self, bpe_path: str = _BPE_PATH):
        self.byte_encoder = _bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(m.split()) for m in merges]
        vocab = list(_bytes_to_unicode().values())
        vocab = vocab + [v + '</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>',
                      '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + '</w>',)
        pairs = _get_pairs(word)
        if not pairs:
            return token + '</w>'
        while True:
            bigram = min(pairs, key=lambda p: self.bpe_ranks.get(p, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = _whitespace_clean(_basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[t] for t in self.bpe(token).split(' '))
        return bpe_tokens


_TOKENIZER = _SimpleTokenizer()


def _tokenize(texts: Union[str, List[str]], context_length: int = 77,
              truncate: bool = False) -> torch.IntTensor:
    """OpenAI CLIP tokenize: pad/truncate to context_length with SOT/EOT."""
    if isinstance(texts, str):
        texts = [texts]
    sot = _TOKENIZER.encoder["<|startoftext|>"]
    eot = _TOKENIZER.encoder["<|endoftext|>"]
    all_tokens = [[sot] + _TOKENIZER.encode(t) + [eot] for t in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot
            else:
                raise RuntimeError(
                    f"Input {texts[i]!r} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result


# =============================================================================
# Inlined CLIP ViT model (from OpenAI CLIP model.py — ViT branch only)
# =============================================================================
# The AesCLIP checkpoints ship as OpenAI-format state_dicts on top of a
# ViT-B/16 backbone, so the ResNet (RN50…) code path from the original
# ``clip/model.py`` is intentionally omitted here.


class _LayerNorm(nn.LayerNorm):
    """LayerNorm that computes in fp32 to stay stable under fp16 inputs."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class _QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class _ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = _LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", _QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = _LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class _Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[
            _ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class _VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = _LayerNorm(width)
        self.transformer = _Transformer(width, layers, heads)
        self.ln_post = _LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)                          # [*, W, H/P, W/P]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [*, W, N]
        x = x.permute(0, 2, 1)                     # [*, N, W]
        x = torch.cat([
            self.class_embedding.to(x.dtype)
            + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
            x,
        ], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # AesCLIP's downstream code multiplies with `self.visual.proj`
        # explicitly, so we return the pre-projection CLS token — matches the
        # upstream VisionTransformer.forward exactly.
        x = self.ln_post(x[:, 0, :])
        return x


class _CLIPViT(nn.Module):
    """Minimal ViT-only CLIP backbone (drop-in for OpenAI CLIP state_dicts)."""

    def __init__(self, embed_dim: int, image_resolution: int, vision_layers: int,
                 vision_width: int, vision_patch_size: int, context_length: int,
                 vocab_size: int, transformer_width: int, transformer_heads: int,
                 transformer_layers: int):
        super().__init__()
        self.context_length = context_length
        vision_heads = vision_width // 64
        self.visual = _VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim,
        )
        self.transformer = _Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self._build_attention_mask(),
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = _LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def _build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x


def _convert_weights_to_fp16(model: nn.Module):
    def _convert(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ("in", "q", "k", "v")],
                         "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ("text_projection", "proj"):
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()
    model.apply(_convert)


def _build_clip_from_state_dict(state_dict: dict) -> nn.Module:
    """Build the ViT-B/16 backbone from an OpenAI CLIP state_dict."""
    assert "visual.proj" in state_dict, \
        "AesCLIP only ships ViT backbones; RN50 checkpoints are not supported."
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([k for k in state_dict
                         if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict
                                 if k.startswith("transformer.resblocks")))

    model = _CLIPViT(
        embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
    )
    for key in ("input_resolution", "context_length", "vocab_size"):
        state_dict.pop(key, None)
    _convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)
    return model.eval()


def _clip_transform(n_px: int):
    return Compose([
        Resize(n_px, interpolation=_BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


def _clip_load(local_pt_path: str, device):
    """Load a CLIP ViT backbone from a local OpenAI-format .pt file.

    Returns ``(model, preprocess)`` in the same shape as ``clip.load()`` from
    the OpenAI reference implementation.
    """
    if not os.path.isfile(local_pt_path):
        raise FileNotFoundError(local_pt_path)
    with open(local_pt_path, 'rb') as f:
        try:
            model = torch.jit.load(f, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            f.seek(0)
            state_dict = torch.load(f, map_location="cpu", weights_only=False)
    model = _build_clip_from_state_dict(state_dict).to(device)
    if str(device) == "cpu":
        model.float()
    return model, _clip_transform(model.visual.input_resolution)


# =============================================================================
# AesCLIP heads (from AesCLIP-main/models/aesclip.py)
# =============================================================================


def _load_clip_backbone(clip_name: str, device):
    """Match ``load_clip_backbone`` from AesCLIP-main.

    Only the local-file (ViT) path is supported here.
    """
    if not os.path.isfile(clip_name):
        raise IOError(
            f"AesCLIP backbone must be a local ViT .pt file; got {clip_name!r}."
        )
    model, _ = _clip_load(clip_name, device)
    return model, 768   # ViT-B/16 feature size


class AesCLIP_reg(nn.Module):
    """IAA regression head on top of a ViT-B/16 CLIP backbone."""

    def __init__(self, clip_name, weight):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, feat_size = _load_clip_backbone(clip_name, self.device)
        print('Loading AesCLIP weights:', clip_model.load_state_dict(torch.load(weight)))
        self.aesclip = clip_model.float()
        self.clip_size = feat_size
        self.mlp = nn.Sequential(
            nn.Linear(self.clip_size, 10),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        img_embedding = self.aesclip.visual(x)
        return self.mlp(img_embedding)


class zs_AesCLIP(nn.Module):
    """Zero-shot AesCLIP scorer — similarity to a pair of text prompts."""

    def __init__(self, clip_name, weight):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, _ = _load_clip_backbone(clip_name, self.device)
        print('Loading AesCLIP weights:', clip_model.load_state_dict(torch.load(weight)))
        self.clip_model = clip_model.float()

    def forward(self, x, texts):
        with torch.no_grad():
            img_embedding = self.clip_model.visual(x)
            img_embedding = img_embedding @ self.clip_model.visual.proj
            text_tokens = torch.cat([_tokenize(t) for t in texts])
            text_embedding = self.clip_model.encode_text(text_tokens.to(self.device)).float()
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            similarity = (100.0 * img_embedding @ text_embedding.T).softmax(dim=-1)
            return [item[0] for item in similarity]


# =============================================================================
# Public scoring API (unchanged interface — CLI compat with the old aes.py)
# =============================================================================


def build_model(mode, device):
    if mode == 'zs':
        model = zs_AesCLIP(clip_name=CLIP_BACKBONE, weight=AESCLIP_WEIGHT)
    else:
        model = AesCLIP_reg(clip_name=CLIP_BACKBONE, weight=AESCLIP_WEIGHT)
        print('Loading IAA weights:',
              model.load_state_dict(torch.load(IAA_WEIGHT, map_location='cpu')))
    return model.to(device).eval()


def score_batch(model, images, mode, score_w):
    """Return a 1D tensor of aesthetic scores on CPU."""
    if mode == 'zs':
        scores = torch.stack(model(images, ['good image', 'bad image'])).float() * 10
    else:
        dist = model(images).float()          # (B, 10) softmax over 1..10
        scores = (dist * score_w).sum(dim=1)  # weighted mean → AVA-style score
    return scores.detach().cpu()


# ---------------------------------------------------------------------------
# Folder dataset
# ---------------------------------------------------------------------------


class ImageFolderFlat(Dataset):
    """Recursively collect images under `root` — returns (tensor, rel_path)."""

    def __init__(self, root, preprocess):
        self.root = root
        self.preprocess = preprocess
        self.paths = []
        for dirpath, _, filenames in os.walk(root):
            for name in sorted(filenames):
                if name.lower().endswith(IMG_EXTS):
                    self.paths.append(os.path.join(dirpath, name))
        self.paths.sort()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        rel = os.path.relpath(path, self.root)
        try:
            with Image.open(path) as img:
                tensor = self.preprocess(img.convert('RGB'))
        except Exception as e:
            print(f'[skip] {rel}: {e}', file=sys.stderr)
            return None
        return tensor, rel


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, []
    tensors, rels = zip(*batch)
    return torch.stack(tensors), list(rels)


def score_image(image_path, mode='iaa', device=None):
    """Score a single image and return a float."""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = build_model(mode, device)
    _, preprocess = _clip_load(CLIP_BACKBONE, device)
    score_w = torch.linspace(1, 10, 10, device=device)

    with Image.open(image_path) as img:
        tensor = preprocess(img.convert('RGB')).unsqueeze(0).to(device)

    with torch.no_grad():
        s = score_batch(model, tensor, mode, score_w)
    return float(s.item())


def score_folder(img_dir, mode='iaa', device=None, batch_size=64,
                 num_workers=4, out_csv=None):
    """Score every image under `img_dir` recursively. Returns dict of stats."""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    model = build_model(mode, device)
    _, preprocess = _clip_load(CLIP_BACKBONE, device)
    score_w = torch.linspace(1, 10, 10, device=device)

    dataset = ImageFolderFlat(img_dir, preprocess)
    print(f'Found {len(dataset)} images under {img_dir}')

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate,
                        pin_memory=(device.type == 'cuda'))

    writer = None
    fout = None
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
        fout = open(out_csv, 'w', newline='')
        writer = csv.writer(fout)
        writer.writerow(['image', 'aes_score'])

    total, score_sum = 0, 0.0
    pbar = tqdm(total=len(dataset), desc='scoring', unit='img', dynamic_ncols=True)
    with torch.no_grad():
        for images, rels in loader:
            if images is None:
                continue
            x = images.to(device, non_blocking=True)
            scores = score_batch(model, x, mode, score_w).tolist()
            for rel, s in zip(rels, scores):
                if writer is not None:
                    writer.writerow([rel, f'{s:.4f}'])
                score_sum += s
            total += len(rels)
            pbar.update(len(rels))
            pbar.set_postfix(mean=f'{score_sum / max(total, 1):.3f}')
    pbar.close()

    if fout is not None:
        fout.close()

    mean_score = (score_sum / total) if total else float('nan')
    return {'count': total, 'mean': mean_score, 'out_csv': out_csv}


# ---------------------------------------------------------------------------
# List-file dataset (4-column txt: content reference gt pred)
# ---------------------------------------------------------------------------


class _PredListDataset(Dataset):
    def __init__(self, pred_paths, preprocess):
        self.pred_paths = pred_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.pred_paths)

    def __getitem__(self, idx):
        path = self.pred_paths[idx]
        try:
            with Image.open(path) as img:
                tensor = self.preprocess(img.convert('RGB'))
        except Exception as e:
            print(f'[skip] {path}: {e}', file=sys.stderr)
            return None
        return tensor, path


_TARGET_COLS = {'content': 0, 'reference': 1, 'gt': 2, 'pred': 3}


def _read_list(list_path, col_idx=3):
    preds = []
    list_dir = os.path.dirname(os.path.abspath(list_path))
    with open(list_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            path = parts[col_idx]
            if not os.path.isabs(path):
                path = os.path.normpath(os.path.join(list_dir, path))
            preds.append(path)
    return preds


def score_list(list_path, out_path, mode='iaa', device=None,
               batch_size=64, num_workers=4, target='pred'):
    """Score the pred column of a 4-column list file and write a txt report."""
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    if target not in _TARGET_COLS:
        raise ValueError(f'target must be one of {list(_TARGET_COLS)}, got {target!r}')
    col_idx = _TARGET_COLS[target]
    pred_paths = _read_list(list_path, col_idx=col_idx)
    print(f'read {len(pred_paths)} {target} paths (column {col_idx}) from {list_path}')

    model = build_model(mode, device)
    _, preprocess = _clip_load(CLIP_BACKBONE, device)
    score_w = torch.linspace(1, 10, 10, device=device)

    dataset = _PredListDataset(pred_paths, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate,
                        pin_memory=(device.type == 'cuda'))

    results = []  # list of (pred_path, score)
    pbar = tqdm(total=len(dataset), desc='scoring', unit='img', dynamic_ncols=True)
    with torch.no_grad():
        for images, paths in loader:
            if images is None:
                continue
            x = images.to(device, non_blocking=True)
            scores = score_batch(model, x, mode, score_w).tolist()
            for path, s in zip(paths, scores):
                results.append((path, s))
            pbar.update(len(paths))
            if results:
                cur_mean = sum(v for _, v in results) / len(results)
                pbar.set_postfix(mean=f'{cur_mean:.3f}')
    pbar.close()

    mean = (sum(v for _, v in results) / len(results)) if results else float('nan')

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(f'# metric: aes ({mode}, AesCLIP)\n')
        f.write(f'# list: {list_path}\n')
        f.write(f'# target column: {target}\n')
        f.write(f'# images: {len(results)}\n')
        for path, s in results:
            f.write(f'{path} {s:.6f}\n')
        f.write(f'AVERAGE {mean:.6f}\n')
    print(f'wrote {out_path}')
    print(f'Scored {len(results)} images, mean aesthetic score: {mean:.4f}')
    return {'count': len(results), 'mean': mean, 'out': out_path}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description='AesCLIP aesthetic scoring.')
    parser.add_argument('--image', type=str, default=None, help='Score a single image.')
    parser.add_argument('--img-dir', type=str, default=None,
                        help='Recursively score all images in a folder.')
    parser.add_argument('--list', type=str, default=None,
                        help='4-column txt list: content reference gt pred (scores the pred column by default).')
    parser.add_argument('--target', choices=list(_TARGET_COLS), default='pred',
                        help='With --list, which column to score (default: pred).')
    parser.add_argument('--out', type=str, default=None,
                        help='Output path. For --img-dir this is a CSV; for --list this is a txt.')
    parser.add_argument('--mode', choices=['zs', 'iaa'], default='iaa',
                        help='iaa: AVA-style 1-10 regression score; '
                             'zs: zero-shot good/bad similarity ×10.')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    picked = sum(x is not None for x in (args.image, args.img_dir, args.list))
    if picked != 1:
        parser.error('Provide exactly ONE of --image, --img-dir, --list.')

    if args.list is not None:
        if args.out is None:
            parser.error('--out is required with --list.')
        if not os.path.isfile(args.list):
            raise FileNotFoundError(args.list)
        score_list(args.list, args.out, mode=args.mode, device=args.device,
                   batch_size=args.batch_size, num_workers=args.num_workers,
                   target=args.target)
        return

    if args.image is not None:
        if not os.path.isfile(args.image):
            raise FileNotFoundError(args.image)
        s = score_image(args.image, mode=args.mode, device=args.device)
        print(f'Aesthetic score ({args.mode}): {s:.4f}')
    else:
        if not os.path.isdir(args.img_dir):
            raise FileNotFoundError(args.img_dir)
        res = score_folder(args.img_dir, mode=args.mode, device=args.device,
                           batch_size=args.batch_size, num_workers=args.num_workers,
                           out_csv=args.out)
        if res['count']:
            print(f"Scored {res['count']} images, mean aesthetic score: {res['mean']:.4f}")
        if res['out_csv']:
            print(f"Results saved to {res['out_csv']}")


if __name__ == '__main__':
    main()
