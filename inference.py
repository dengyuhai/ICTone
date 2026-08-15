from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import FluxFillPipeline
from PIL import Image


DEFAULT_INSTANCE_PROMPT = (
    "A side-by-side triptych. Left: source photo. "
    "Middle: a color and tone reference photo. "
    "Right: the same scene as the left, re-graded so its colors, "
    "contrast, and film look match the middle reference, while "
    "preserving the left's content and details."
)


def _resize_to_wh(pil: Image.Image, width: int, height: int) -> Image.Image:
    """Aspect-preserving resize-cover then center-crop to (width, height)."""
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    w, h = pil.size
    s = max(width / w, height / h)
    nw = max(int(round(w * s)), width)
    nh = max(int(round(h * s)), height)
    pil = pil.resize((nw, nh), Image.LANCZOS)
    left = (nw - width) // 2
    top = (nh - height) // 2
    return pil.crop((left, top, left + width, top + height))


def _resize_width_keep_aspect(
    pil: Image.Image, width: int, height_multiple: int = 16
) -> Image.Image:
    """Resize so output width == ``width``, height scales by original aspect.

    Height is rounded to the nearest positive multiple of ``height_multiple``
    (FluxFill's VAE + patch stride, so the pipeline accepts it without extra
    padding). No cropping is performed.
    """
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    w, h = pil.size
    new_h = int(round(h * width / w))
    if height_multiple > 1:
        new_h = int(round(new_h / height_multiple)) * height_multiple
        new_h = max(height_multiple, new_h)
    else:
        new_h = max(1, new_h)
    return pil.resize((width, new_h), Image.LANCZOS)


def _build_lut_laplacian(size: int):
    """Sparse 3D graph-Laplacian on the (size, size, size) LUT grid.

    Each vertex has up to 6 axis-aligned neighbors. ``L L^T`` acts as a
    curvature penalty so uncovered / sparsely-covered LUT cells extrapolate
    smoothly from their neighbors instead of collapsing to a fixed anchor.
    """
    from scipy.sparse import coo_matrix
    V = size ** 3
    idx = np.arange(V, dtype=np.int64).reshape(size, size, size)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for axis in range(3):
        # Pair each vertex with its +1-neighbor along ``axis`` (no wrap-around).
        take = [slice(None)] * 3
        take[axis] = slice(None, -1)
        src = idx[tuple(take)].ravel()
        take[axis] = slice(1, None)
        dst = idx[tuple(take)].ravel()
        # Edge (src -> dst): L[src] - L[dst] = 0 (finite difference row).
        n = src.shape[0]
        edge_rows = np.arange(n, dtype=np.int64) + sum(r.shape[0] for r in rows) // 2 * 0  # local re-baseline below
        # We'll assemble a single edge-per-row Laplacian by concatenating below.
        rows.append(src)
        cols.append(src)
        data.append(np.ones(n, dtype=np.float32))
        rows.append(src)
        cols.append(dst)
        data.append(-np.ones(n, dtype=np.float32))
        rows.append(dst)
        cols.append(dst)
        data.append(np.ones(n, dtype=np.float32))
        rows.append(dst)
        cols.append(src)
        data.append(-np.ones(n, dtype=np.float32))
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    d = np.concatenate(data)
    # This is a graph Laplacian L in vertex-index form; ``LtL`` == L (since L is
    # symmetric PSD). Return it as the penalty operator directly.
    return coo_matrix((d, (r, c)), shape=(V, V)).tocsr()


def estimate_lut(
    before,
    after,
    size: int = 33,
    lam_smooth: float = 0.05,
    lam_anchor: float = 1e-5,
    device=None,
):
    """Estimate a size^3 3D LUT mapping ``before`` colors to ``after`` colors.

    Trilinear-consistent sparse least squares with a **Laplacian smoothness
    prior** and a tiny identity anchor to break gauge invariance:

        min_L  ||W L - after||^2
               + lam_smooth * L^T Δ L                  (3D grid smoothness)
               + lam_anchor * ||L - identity||^2       (gauge fix)

    The smoothness prior is the key fix for tone-migration artifacts (e.g.
    saturated colors like lips): sparsely-covered LUT cells extrapolate from
    neighboring *covered* cells that carry the correct color transform,
    instead of being pulled back toward the input color by an identity anchor.

    ``before`` / ``after`` may be HxWx3 arrays or flat (N, 3) arrays with the
    same length. ``device`` picks the solver backend:
      - None / "cpu": scipy sparse direct solve (spsolve).
      - "cuda" / torch.device("cuda"): torch sparse block-CG on GPU (~90x
        faster on hi-res pairs).
    Falls back to nearest-cell averaging if SciPy is unavailable on CPU path.
    """
    if device is not None and str(device) != "cpu":
        return _estimate_lut_gpu(
            before, after, size, lam_smooth, lam_anchor, device=device
        )
    return _estimate_lut_cpu(before, after, size, lam_smooth, lam_anchor)


def _estimate_lut_cpu(before, after, size, lam_smooth, lam_anchor):
    before = np.asarray(before, dtype=np.float32).reshape(-1, 3)
    after = np.asarray(after, dtype=np.float32).reshape(-1, 3)
    assert before.shape == after.shape, "before/after must have same shape"

    V = size ** 3

    # Identity LUT anchor (used by both the LS solve and the fallback).
    grid = np.linspace(0, 255, size, dtype=np.float32)
    R, G, B = np.meshgrid(grid, grid, grid, indexing='ij')
    identity_flat = np.stack([R, G, B], axis=-1).reshape(V, 3)

    try:
        from scipy.sparse import coo_matrix, eye as sp_eye
        from scipy.sparse.linalg import spsolve
    except ImportError:
        return _estimate_lut_nearest_fallback(
            before, after, size, identity_flat
        ).reshape(size, size, size, 3)

    # Continuous LUT-grid coordinates per pixel.
    pos = before / 255.0 * (size - 1)
    i0 = np.clip(np.floor(pos).astype(np.int64), 0, size - 1)
    i1 = np.clip(i0 + 1, 0, size - 1)
    f = pos - i0
    fr, fg, fb = f[:, 0], f[:, 1], f[:, 2]
    ofr, ofg, ofb = 1.0 - fr, 1.0 - fg, 1.0 - fb
    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    r1, g1, b1 = i1[:, 0], i1[:, 1], i1[:, 2]

    def _vidx(r, g, b):
        return (r * size + g) * size + b

    verts = np.stack([
        _vidx(r0, g0, b0), _vidx(r0, g0, b1),
        _vidx(r0, g1, b0), _vidx(r0, g1, b1),
        _vidx(r1, g0, b0), _vidx(r1, g0, b1),
        _vidx(r1, g1, b0), _vidx(r1, g1, b1),
    ], axis=1)
    wts = np.stack([
        ofr * ofg * ofb, ofr * ofg * fb,
        ofr * fg * ofb, ofr * fg * fb,
        fr * ofg * ofb, fr * ofg * fb,
        fr * fg * ofb, fr * fg * fb,
    ], axis=1).astype(np.float32)

    N = before.shape[0]
    rows = np.repeat(np.arange(N, dtype=np.int64), 8)
    cols = verts.reshape(-1)
    data = wts.reshape(-1)
    W = coo_matrix((data, (rows, cols)), shape=(N, V)).tocsr()

    WtW = (W.T @ W).tocsc()
    scale = max(float(WtW.diagonal().mean()), 1.0)
    lap = _build_lut_laplacian(size).tocsc()
    A = (WtW + (lam_smooth * scale) * lap + lam_anchor * sp_eye(V, format='csc')).tocsc()
    rhs = W.T @ after + lam_anchor * identity_flat

    lut_flat = np.empty((V, 3), dtype=np.float32)
    for c in range(3):
        lut_flat[:, c] = spsolve(A, rhs[:, c])

    return lut_flat.reshape(size, size, size, 3)


def _estimate_lut_gpu(
    before, after, size, lam_smooth, lam_anchor, device,
    tol: float = 1e-4, max_iter: int = 200,
):
    """GPU LS solve via preconditioned block conjugate gradient.

    Solves the same system as ``_estimate_lut_cpu`` but on the specified CUDA
    device using ``torch.sparse.mm`` for matvecs. Returns a numpy array shaped
    ``(size, size, size, 3)`` matching the CPU path.
    """
    b = torch.as_tensor(before, dtype=torch.float32, device=device).reshape(-1, 3)
    y = torch.as_tensor(after, dtype=torch.float32, device=device).reshape(-1, 3)
    N = b.shape[0]
    V = size ** 3

    pos = b / 255.0 * (size - 1)
    i0 = pos.floor().clamp(0, size - 1).long()
    i1 = (i0 + 1).clamp(0, size - 1)
    f = pos - i0.float()
    fr, fg, fb = f[:, 0], f[:, 1], f[:, 2]
    ofr, ofg, ofb = 1 - fr, 1 - fg, 1 - fb
    r0, g0, b0 = i0[:, 0], i0[:, 1], i0[:, 2]
    r1, g1, b1 = i1[:, 0], i1[:, 1], i1[:, 2]

    def _vidx(r, g, c):
        return (r * size + g) * size + c

    verts = torch.stack([
        _vidx(r0, g0, b0), _vidx(r0, g0, b1),
        _vidx(r0, g1, b0), _vidx(r0, g1, b1),
        _vidx(r1, g0, b0), _vidx(r1, g0, b1),
        _vidx(r1, g1, b0), _vidx(r1, g1, b1),
    ], dim=1)
    wts = torch.stack([
        ofr * ofg * ofb, ofr * ofg * fb,
        ofr * fg * ofb, ofr * fg * fb,
        fr * ofg * ofb, fr * ofg * fb,
        fr * fg * ofb, fr * fg * fb,
    ], dim=1)

    rows = torch.arange(N, device=device).repeat_interleave(8)
    cols = verts.reshape(-1)
    vals = wts.reshape(-1)
    W = torch.sparse_coo_tensor(torch.stack([rows, cols]), vals, (N, V)).coalesce()
    Wt = W.transpose(0, 1).coalesce()

    # 3D graph Laplacian on the LUT grid.
    idx3 = torch.arange(V, device=device).view(size, size, size)
    L_rows: list[torch.Tensor] = []
    L_cols: list[torch.Tensor] = []
    L_vals: list[torch.Tensor] = []
    for ax in range(3):
        sl = [slice(None)] * 3
        sl[ax] = slice(None, -1)
        src = idx3[tuple(sl)].reshape(-1)
        sl[ax] = slice(1, None)
        dst = idx3[tuple(sl)].reshape(-1)
        ones_s = torch.ones_like(src, dtype=torch.float32)
        ones_d = torch.ones_like(dst, dtype=torch.float32)
        L_rows += [src, dst, src, dst]
        L_cols += [src, dst, dst, src]
        L_vals += [ones_s, ones_d, -ones_s, -ones_d]
    Lop = torch.sparse_coo_tensor(
        torch.stack([torch.cat(L_rows), torch.cat(L_cols)]),
        torch.cat(L_vals), (V, V)).coalesce()

    # WᵀW diag + Laplacian diag → Jacobi preconditioner.
    WtW_diag = torch.zeros(V, device=device).scatter_add_(
        0, verts.reshape(-1), wts.reshape(-1) ** 2)
    scale = max(float(WtW_diag.mean().item()), 1.0)
    lam_s = lam_smooth * scale
    lam_a = lam_anchor
    L_diag = torch.zeros(V, device=device)
    same = Lop.indices()[0] == Lop.indices()[1]
    L_diag.scatter_add_(0, Lop.indices()[0][same], Lop.values()[same])
    Minv = 1.0 / (WtW_diag + lam_s * L_diag + lam_a)

    grid = torch.linspace(0, 255, size, device=device)
    R, G, B = torch.meshgrid(grid, grid, grid, indexing='ij')
    identity = torch.stack([R, G, B], dim=-1).reshape(V, 3)

    def A_matvec(X):
        return (torch.sparse.mm(Wt, torch.sparse.mm(W, X))
                + lam_s * torch.sparse.mm(Lop, X)
                + lam_a * X)

    rhs = torch.sparse.mm(Wt, y) + lam_a * identity

    # Preconditioned block Conjugate Gradient (per-channel in parallel).
    X = torch.zeros_like(rhs)
    R_ = rhs - A_matvec(X)
    Z = Minv.unsqueeze(1) * R_
    P = Z.clone()
    rz_old = (R_ * Z).sum(dim=0)
    b_norm = rhs.norm(dim=0).clamp_min(1e-30)
    for _ in range(max_iter):
        AP = A_matvec(P)
        alpha = rz_old / ((P * AP).sum(dim=0) + 1e-30)
        X = X + alpha.unsqueeze(0) * P
        R_ = R_ - alpha.unsqueeze(0) * AP
        if (R_.norm(dim=0) / b_norm).max().item() < tol:
            break
        Z = Minv.unsqueeze(1) * R_
        rz_new = (R_ * Z).sum(dim=0)
        P = Z + (rz_new / rz_old).unsqueeze(0) * P
        rz_old = rz_new

    return X.reshape(size, size, size, 3).detach().cpu().numpy()


def _estimate_lut_nearest_fallback(before, after, size, identity_flat):
    """Old nearest-cell averaging path, used only if SciPy is missing."""
    idx = np.clip(
        np.round(before / 255.0 * (size - 1)).astype(np.int32),
        0, size - 1)
    flat = (idx[:, 0] * size + idx[:, 1]) * size + idx[:, 2]

    V = size ** 3
    lut = np.zeros((V, 3), dtype=np.float32)
    counts = np.zeros(V, dtype=np.int64)
    np.add.at(lut, flat, after)
    np.add.at(counts, flat, 1)
    filled = counts > 0
    lut[filled] /= counts[filled, None]
    lut[~filled] = identity_flat[~filled]
    return lut


def apply_lut(content, lut, device=None):
    """Apply a 3D LUT to an RGB image (any resolution) with trilinear interp.

    ``device=None`` runs the numpy path (portable); a CUDA device runs it via
    ``torch.nn.functional.grid_sample`` (~50x faster on hi-res images).
    """
    if device is not None and str(device) != "cpu":
        return _apply_lut_gpu(content, lut, device)
    return _apply_lut_cpu(content, lut)


def _apply_lut_cpu(content, lut):
    size = lut.shape[0]
    img = np.asarray(content, dtype=np.float32) / 255.0 * (size - 1)

    i0 = np.floor(img).astype(np.int32)
    i0 = np.clip(i0, 0, size - 1)
    i1 = np.clip(i0 + 1, 0, size - 1)
    f = img - i0

    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr = f[..., 0:1]
    fg = f[..., 1:2]
    fb = f[..., 2:3]

    c000 = lut[r0, g0, b0]
    c001 = lut[r0, g0, b1]
    c010 = lut[r0, g1, b0]
    c011 = lut[r0, g1, b1]
    c100 = lut[r1, g0, b0]
    c101 = lut[r1, g0, b1]
    c110 = lut[r1, g1, b0]
    c111 = lut[r1, g1, b1]

    c00 = c000 * (1 - fb) + c001 * fb
    c01 = c010 * (1 - fb) + c011 * fb
    c10 = c100 * (1 - fb) + c101 * fb
    c11 = c110 * (1 - fb) + c111 * fb
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    out = c0 * (1 - fr) + c1 * fr

    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_lut_gpu(content, lut, device):
    """GPU trilinear LUT application via ``F.grid_sample`` (5D volume)."""
    import torch.nn.functional as F
    size = lut.shape[0]
    lut_np = np.asarray(lut, dtype=np.float32)
    lut_t = torch.from_numpy(lut_np).permute(3, 0, 1, 2).unsqueeze(0).to(device)
    # LUT dim layout after permute: (N=1, C=3, D=R, H=G, W=B).

    img = torch.from_numpy(np.asarray(content, dtype=np.float32)).to(device)
    coord = img / 255.0 * (size - 1)
    coord = 2 * coord / (size - 1) - 1                # → [-1, 1] w.r.t. (R, G, B)
    # grid_sample expects last-dim order (x, y, z) == (W, H, D) == (B, G, R).
    coord = coord[..., [2, 1, 0]].unsqueeze(0).unsqueeze(0)   # (1, 1, H, W, 3)
    out = F.grid_sample(
        lut_t, coord, mode="bilinear", align_corners=True, padding_mode="border"
    )
    out = out.squeeze(0).squeeze(1).permute(1, 2, 0)
    return out.clamp(0, 255).byte().cpu().numpy()


def build_triptych(
    content: Image.Image, reference: Image.Image, panel_w: int, panel_h: int
):
    """Return (triptych_pil, mask_pil) both at (3*panel_w, panel_h).

    - Triptych: ``[content | reference | content]`` — the right third is a
      copy of content, acting as an identity prior for prepare_latents
      (matches the training/validation convention).
    - Mask: L-mode PIL, 0 elsewhere and 255 on the right third (fill region).
    """
    canvas_w = panel_w * 3
    canvas_h = panel_h

    tri = Image.new("RGB", (canvas_w, canvas_h))
    tri.paste(content, (0, 0))
    tri.paste(reference, (panel_w, 0))
    tri.paste(content, (2 * panel_w, 0))

    mask_arr = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask_arr[:, 2 * panel_w:] = 255
    mask = Image.fromarray(mask_arr, mode="L")
    return tri, mask


def run_one(
    pipe: FluxFillPipeline,
    content_pil: Image.Image,
    reference_pil: Image.Image,
    *,
    size: int,
    prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    generator_device: str,
) -> tuple[Image.Image, Image.Image, Image.Image, Image.Image]:
    """Run one triptych inference. Returns (pred_pil, panel_pil, tri_pil, content_sq).

    - ``pred_pil``: just the right-third region — the migrated result.
    - ``panel_pil``: horizontal panel ``[content | reference | pred]``.
    - ``tri_pil``: the full pipeline output. Useful for debugging.
    - ``content_sq``: the resized content actually fed to the pipeline. Kept
      so callers can pair it with ``pred_pil`` for LUT fitting.

    Content's width is resized to ``size`` while its height keeps the original
    aspect ratio (rounded to a multiple of 16 for the VAE). Reference is
    force-resized to the same ``(size, H)`` so the triptych panels line up.
    Canvas is ``3*size x H``.
    """
    content = _resize_width_keep_aspect(content_pil, size, height_multiple=16)
    panel_w, panel_h = content.size
    # Stretch reference directly to content's (W, H) — no aspect preservation,
    # no cropping. Panels line up by construction.
    if reference_pil.mode != "RGB":
        reference_pil = reference_pil.convert("RGB")
    reference = reference_pil.resize((panel_w, panel_h), Image.LANCZOS)

    tri, mask = build_triptych(content, reference, panel_w, panel_h)

    generator = torch.Generator(device=generator_device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        image=tri,
        mask_image=mask,
        height=panel_h,
        width=panel_w * 3,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        max_sequence_length=512,
        generator=generator,
    ).images[0]

    pred = result.crop((panel_w * 2, 0, panel_w * 3, panel_h))
    panel = Image.new("RGB", (panel_w * 3, panel_h))
    panel.paste(content, (0, 0))
    panel.paste(reference, (panel_w, 0))
    panel.paste(pred, (panel_w * 2, 0))
    return pred, panel, result, content


def main():
    parser = argparse.ArgumentParser("Zero-shot tone migration with FLUX.1-Fill")
    # Single-pair inputs
    parser.add_argument("--content", type=str, default=None,
                        help="Path to the content (source) image.")
    parser.add_argument("--reference", type=str, default=None,
                        help="Path to the reference (style/tone) image.")
    # Batch mode over TST2K
    parser.add_argument("--tst2k-dir", type=str, default=None,
                        help="Root of TST2K-style eval set. Each subdir must contain "
                             "content.png + reference.png. If set, --content/--reference "
                             "are ignored and up to --tst2k-num subdirs are processed.")
    parser.add_argument("--triplet-list", type=str, default=None,
                        help="Text file with 3 whitespace-separated columns per row: "
                             "<content_path> <reference_path> <gt_path>. Comment lines "
                             "starting with '#' are skipped. Overrides --tst2k-dir and "
                             "--content/--reference. Row index (0-based) becomes the "
                             "output stem (NNNN.png).")
    parser.add_argument("--tst2k-num", type=int, default=50,
                        help="Max number of samples (subdirs or list rows) to iterate.")

    parser.add_argument("--output-dir", type=str, default="./tone_out",
                        help="Directory for batch outputs. Ignored when --output-file is set.")
    parser.add_argument("--output-file", type=str, default=None,
                        help="Exact output image path for single-pair inference. "
                             "When set, --output-dir is not used.")
    parser.add_argument("--flux-path", type=str,
                        default="ckpt/FLUX-Fill")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="Optional LoRA path (dir or .safetensors). Omit to test "
                             "the base FluxFill model with no fine-tuning.")

    parser.add_argument("--image-size", type=int, default=512,
                        help="Content width; height keeps the source aspect "
                             "ratio (rounded to a multiple of 16 for the VAE). "
                             "Canvas width = 3 * this.")
    parser.add_argument("--lut-size", type=int, default=33,
                        help="3D LUT grid size per channel used to lift the "
                             "512-res migration back onto the original hi-res "
                             "content. Set to 0 to skip hi-res reconstruction.")
    parser.add_argument("--lut-device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Solver device for LUT estimate + apply. 'auto' "
                             "uses CUDA when available (~90x faster estimate, "
                             "~50x faster apply on hi-res images).")
    parser.add_argument("--num-inference-steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=30.0,
                        help="FluxFill's guidance-distilled embed value. ICEdit's "
                             "reference inference.py uses 50; 30 is FluxFill default.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default=None,
                        help="Text prompt. Omit to use the built-in triptych "
                             "instruction (identical to the training default).")

    parser.add_argument("--enable-model-cpu-offload", action="store_true")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--generator-device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Where the noise generator lives. ICEdit uses 'cpu'; "
                             "'cuda' silences the diffusers 'passed generator was "
                             "created on cpu' warning at the cost of slightly "
                             "different bit-exact noise across restarts.")

    # Data-parallel sharding across independent processes (one per GPU).
    # Each shard iterates the same global job list but only processes indices
    # ``i`` where ``i % num_shards == shard_index``. The per-sample seed is
    # derived from the *global* index so results are identical to a
    # single-process run.
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of parallel shards (processes).")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="This shard's index in [0, num_shards).")

    args = parser.parse_args()

    if args.triplet_list is None and args.tst2k_dir is None \
            and (args.content is None or args.reference is None):
        parser.error(
            "Provide one of: --triplet-list, --tst2k-dir, or both --content and --reference."
        )

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error(
            f"Invalid sharding: shard_index={args.shard_index}, "
            f"num_shards={args.num_shards}."
        )

    if args.output_file is not None and (
        args.triplet_list is not None or args.tst2k_dir is not None
    ):
        parser.error("--output-file is only supported for single-pair inference.")

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"[load] FluxFill from {args.flux_path}  dtype={args.dtype}")
    pipe = FluxFillPipeline.from_pretrained(args.flux_path, torch_dtype=torch_dtype)

    if args.lora_path:
        print(f"[load] LoRA weights from {args.lora_path}")
        pipe.load_lora_weights(args.lora_path)
    else:
        print("[load] no LoRA — testing base FluxFill zero-shot")

    if args.enable_model_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    prompt = args.prompt if args.prompt is not None else DEFAULT_INSTANCE_PROMPT
    print(f"[prompt] {prompt}")

    if args.lut_device == "auto":
        lut_device = "cuda" if torch.cuda.is_available() else None
    elif args.lut_device == "cuda":
        lut_device = "cuda"
    else:
        lut_device = None
    print(f"[lut] solver device: {lut_device or 'cpu'}")

    output_file = Path(args.output_file) if args.output_file is not None else None
    if output_file is not None:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        dir_out = str(output_file.parent)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        dir_out = os.path.join(args.output_dir, "outputs")
        os.makedirs(dir_out, exist_ok=True)
    S = int(args.image_size)

    # ---- Assemble list of (stem, content_path, reference_path, gt_path) ----
    jobs: list[tuple[str, Path, Path, Path | None]] = []
    if args.triplet_list is not None:
        list_path = Path(args.triplet_list)
        list_base = list_path.resolve().parent

        def resolve_list_path(value: str) -> Path:
            path = Path(value)
            return path if path.is_absolute() else list_base / path

        with open(list_path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                c = resolve_list_path(parts[0])
                r = resolve_list_path(parts[1])
                g = resolve_list_path(parts[2]) if len(parts) >= 3 else None
                if not (c.exists() and r.exists()):
                    continue
                if g is not None and not g.exists():
                    g = None
                stem = f"{len(jobs):04d}"
                jobs.append((stem, c, r, g))
                if len(jobs) >= args.tst2k_num:
                    break
        if not jobs:
            raise SystemExit(f"[error] no valid rows in {list_path}")
        print(f"[batch] {len(jobs)} samples from {list_path}")
    elif args.tst2k_dir is not None:
        root = Path(args.tst2k_dir)
        subs = sorted([p for p in root.iterdir() if p.is_dir()])
        for sub in subs:
            c = sub / "content.png"
            r = sub / "reference.png"
            if not (c.exists() and r.exists()):
                continue
            g = sub / "gt.png"
            jobs.append((sub.name, c, r, g if g.exists() else None))
            if len(jobs) >= args.tst2k_num:
                break
        if not jobs:
            raise SystemExit(f"[error] no valid subdirs found under {root}")
        print(f"[batch] {len(jobs)} samples from {root}")
    else:
        c = Path(args.content)
        r = Path(args.reference)
        jobs.append((c.stem, c, r, None))

    # ---- Filter jobs for this shard while keeping the global index ----
    # ``global_i`` is the index into the full (unsharded) job list. It drives
    # both the per-sample seed (``args.seed + global_i``) and the output file
    # prefix, so different shards write disjoint filenames and any single
    # sample gets the same seed regardless of shard configuration.
    total_jobs = len(jobs)
    sharded = [
        (gi, job) for gi, job in enumerate(jobs)
        if gi % args.num_shards == args.shard_index
    ]
    if args.num_shards > 1:
        print(
            f"[shard] {args.shard_index+1}/{args.num_shards}: "
            f"{len(sharded)}/{total_jobs} samples"
        )

    # ---- Run inference ----
    for local_i, (global_i, (stem, cpath, rpath, gpath)) in enumerate(sharded):
        content_pil = Image.open(cpath).convert("RGB")
        reference_pil = Image.open(rpath).convert("RGB")

        pred, panel, full, content_sq = run_one(
            pipe,
            content_pil,
            reference_pil,
            size=S,
            prompt=prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            seed=args.seed + global_i,
            generator_device=args.generator_device,
        )

        # ---- Hi-res reconstruction via 3D LUT ----
        # The pipeline works at S=512, but ``content_pil`` is usually
        # higher-resolution. Fit a 3D LUT so its trilinear evaluation on the
        # hi-res content reproduces the diffusion output.
        #
        # Training pair: hi-res content pixels paired with pred bilinearly
        # upsampled to hi-res. This gives the LUT the exact color distribution
        # it will be applied to (important for saturated regions like lips,
        # whose peak reds get lost when content is first downsampled). We
        # subsample to keep the sparse LS problem small.
        pred_hires = None
        if args.lut_size and args.lut_size > 1:
            content_arr = np.asarray(content_pil)
            pred_up = np.asarray(
                pred.resize(content_pil.size, Image.BILINEAR)
            )
            flat_before = content_arr.reshape(-1, 3)
            flat_after = pred_up.reshape(-1, 3)
            max_samples = 500_000
            if flat_before.shape[0] > max_samples:
                rng = np.random.default_rng(0)
                sel = rng.choice(
                    flat_before.shape[0], size=max_samples, replace=False
                )
                flat_before = flat_before[sel]
                flat_after = flat_after[sel]
            lut = estimate_lut(
                flat_before,
                flat_after,
                size=int(args.lut_size),
                device=lut_device,
            )
            hires_arr = apply_lut(content_arr, lut, device=lut_device)
            pred_hires = Image.fromarray(hires_arr)

        # Content-resolution reconstruction goes to ``outputs/``.
        # Filename is just the (1-based) global index, zero-padded to 4 digits,
        # so shards never collide and results sort naturally.
        code = f"{global_i :04d}"
        name_out = str(output_file) if output_file is not None else str(Path(dir_out) / code)
        if pred_hires is not None:
            pred_hires.save(
                str(output_file) if output_file is not None else f"{name_out}.png"
            )

        print(
            f"[done] shard {args.shard_index+1}/{args.num_shards}  "
            f"{local_i+1}/{len(sharded)}  (global {global_i+1}/{total_jobs})  "
            f"{stem}  →  {name_out}.png"
        )

    print(
        f"[all done] shard {args.shard_index+1}/{args.num_shards} — "
        f"results under {os.path.abspath(args.output_dir)}"
    )


if __name__ == "__main__":
    main()
