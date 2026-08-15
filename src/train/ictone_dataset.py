from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

try:
    from pillow_lut import load_cube_file
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "pillow_lut is required for ICToneDataset. Install with `pip install pillow-lut`."
    ) from _e


logger = logging.getLogger(__name__)


IMAGE_EXTS = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".webp", ".bmp")


DEFAULT_INSTANCE_PROMPT = (
    "A side-by-side triptych. Left: source photo. "
    "Middle: a color and tone reference photo. "
    "Right: the same scene as the left, re-graded so its colors, "
    "contrast, and film look match the middle reference, while "
    "preserving the left's content and details."
)


def _list_source_images(root: str | os.PathLike) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"source dir does not exist: {root}")
    files = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise RuntimeError(f"no images found under {root}")
    return sorted(files)


def _load_lut_list(txt_path: str | os.PathLike) -> list[str]:
    txt_path = Path(txt_path)
    with open(txt_path, "r") as f:
        paths = [ln.strip() for ln in f if ln.strip()]
    if not paths:
        raise RuntimeError(f"empty LUT list: {txt_path}")
    return paths


def _resize_width_keep_aspect(
    pil: Image.Image, width: int, height_multiple: int = 16
) -> Image.Image:
    """Resize so output width == ``width``, height scales by original aspect.

    Height is snapped to a positive multiple of ``height_multiple`` (16 =
    FluxFill's VAE + patch stride) so the trainer / pipeline accepts it
    without extra padding. No cropping is performed.
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


class ICToneDataset(Dataset):
    """Filter-migration triplet dataset for the ICEdit / OminiModel trainer.

    Args mirror the ICEdit dataset classes:
        source_dir: directory of source images (e.g. ``ppr10k_source_360``).
        lut_list_path: text file with one ``.cube`` path per line.
        condition_size / target_size: per-part square size. Both must match
            (kept as two args purely to stay wire-compatible with ICEdit's
            YAML configs).
        drop_text_prob: probability of replacing the prompt with ``" "``.
        instance_prompt: text prompt describing the triptych task. ``None``
            → :data:`DEFAULT_INSTANCE_PROMPT`.
        length: virtual epoch length. ``None`` → ``|sources| * repeats``.
        repeats: virtual epoch multiplier.
    """

    def __init__(
        self,
        source_dir: str | os.PathLike = "data/ppr10k_source_360",
        lut_list_path: str | os.PathLike = "data/ICTone_lut_unique_visual.txt",
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        instance_prompt: str | None = None,
        length: int | None = None,
        repeats: int = 1,
    ) -> None:
        if int(condition_size) != int(target_size):
            raise ValueError(
                "ICToneDataset requires condition_size == target_size; got "
                f"{condition_size} vs {target_size}."
            )
        self.source_paths: list[Path] = _list_source_images(source_dir)
        self.lut_paths: list[str] = _load_lut_list(lut_list_path)
        if len(self.source_paths) < 2:
            raise RuntimeError("need at least 2 source images to form a triplet")
        if len(self.lut_paths) < 2:
            raise RuntimeError("need at least 2 LUTs to form a triplet")
        self.image_size = int(condition_size)
        self.drop_text_prob = float(drop_text_prob)
        self.instance_prompt = instance_prompt if instance_prompt is not None else DEFAULT_INSTANCE_PROMPT
        self.repeats = max(1, int(repeats))
        self._length = int(length) if length is not None else len(self.source_paths) * self.repeats
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return self._length

    # ------------------------------------------------------------------
    def _load_source(self, idx: int) -> Image.Image:
        p = self.source_paths[idx % len(self.source_paths)]
        return Image.open(p).convert("RGB")

    def _load_lut(self, max_retries: int = 8):
        tried: set[int] = set()
        for _ in range(max_retries):
            idx = random.randrange(len(self.lut_paths))
            if idx in tried:
                continue
            tried.add(idx)
            p = self.lut_paths[idx]
            try:
                lut = load_cube_file(p)
                return lut, p
            except Exception as e:  # noqa: BLE001
                logger.debug(f"LUT load failed for {p}: {e}")
                continue
        raise RuntimeError(f"could not load any LUT after {max_retries} attempts")

    def _sample_unique_luts(self, k: int):
        """Sample ``k`` distinct LUTs (by path). Falls back to allowing repeats
        only if the pool is smaller than ``k``."""
        luts: list = []
        paths: list[str] = []
        max_attempts = max(16, k * 8)
        attempts = 0
        while len(luts) < k and attempts < max_attempts:
            attempts += 1
            lut, path = self._load_lut()
            if path in paths and len(self.lut_paths) >= k:
                continue
            luts.append(lut)
            paths.append(path)
        if len(luts) < k:
            raise RuntimeError(f"could not sample {k} LUTs after {max_attempts} attempts")
        return luts, paths

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        n = len(self.source_paths)
        idx_a = random.randrange(n)
        idx_b = random.randrange(n - 1)
        if idx_b >= idx_a:
            idx_b += 1

        for _ in range(8):
            try:
                pil_a_raw = self._load_source(idx_a)
                pil_b_raw = self._load_source(idx_b)
                break
            except Exception:  # noqa: BLE001
                idx_a = random.randrange(n)
                idx_b = random.randrange(n - 1)
                if idx_b >= idx_a:
                    idx_b += 1
        else:
            raise RuntimeError("failed to load any source image pair after retries")

        S = self.image_size
        # Width -> S, height scales by pil_a's aspect (snapped to /16 for
        # the VAE). Reference is force-resized (no crop, no aspect keep)
        # directly to (S, H) so all three triptych panels line up. For tone
        # migration the reference is only used for its global colour / tone
        # distribution, so stretching is acceptable and preserves 100% of
        # the reference pixels.
        pil_a = _resize_width_keep_aspect(pil_a_raw, S, height_multiple=16)
        H = pil_a.size[1]
        pil_b = pil_b_raw.convert("RGB").resize((S, H), Image.LANCZOS)

        # Sample K distinct LUTs where K in [2, 4]:
        #   - luts[0]          -> style + GT
        #   - luts[1:]  (1..3) -> stacked in order to form the content LUT chain
        k = random.randint(2, 4)
        luts, _paths = self._sample_unique_luts(k)
        lut_s = luts[0]
        content_luts = luts[1:]

        def _apply(im: Image.Image, lut) -> Image.Image:
            try:
                out = im.filter(lut)
                if out.mode != "RGB":
                    out = out.convert("RGB")
                return out
            except Exception as e:  # noqa: BLE001
                logger.warning(f"LUT apply failed ({e}); using identity")
                return im.copy()

        def _apply_chain(im: Image.Image, lut_chain) -> Image.Image:
            out = im
            for lut in lut_chain:
                out = _apply(out, lut)
            return out

        content = _apply_chain(pil_a, content_luts)
        style = _apply(pil_b, lut_s)
        gt = _apply(pil_a, lut_s)

        # Triptych target = [content | style | GT], shape (3S, H).
        target = Image.new("RGB", (S * 3, H))
        target.paste(content, (0, 0))
        target.paste(style, (S, 0))
        target.paste(gt, (S * 2, 0))

        # L-mode PIL mask, mirroring ICEdit's EditDataset (Image.new('L', ..., 0)
        # + ImageDraw.rectangle fill=255). The pipeline internally normalises
        # this to [0, 1] and does ``masked_image = image * (1 - mask)``.
        mask = Image.new("L", (S * 3, H), 0)
        draw = ImageDraw.Draw(mask)
        draw.rectangle([S * 2, 0, S * 3, H], fill=255)

        # Prompt with optional dropout (ICEdit convention uses " " for dropped).
        if random.random() < self.drop_text_prob:
            description = " "
        else:
            description = self.instance_prompt

        return {
            "image": self.to_tensor(target),         # (3, S, 3S), [0, 1]
            "condition": self.to_tensor(mask),       # (1, S, 3S), [0, 1]
            "condition_type": "ictone",
            "description": description,
            "position_delta": np.array([0, 0]),
        }


class ICTonePairDataset(Dataset):
    """Pair-driven variant of :class:`ICToneDataset`.

    Source of truth is a merged pair .npz (produced by
    ``TST100K_build/unify_sampled_pairs_A.py``). Each pair provides:

        q_image, q_lut   -> used for the source (pil_a) and the **GT** LUT
        r_image, r_lut   -> used for the reference (pil_b) and the **style** LUT

    Triptych = ``[ content(pil_a, chain) | style(pil_b, r_lut) | gt(pil_a, q_lut) ]``

    The content branch keeps the original random multi-LUT chain: K-1 LUTs
    (K in [2, 4]) sampled from the merged ``lut_index.tsv`` (or an override
    pool) are applied to ``pil_a`` in order.

    Compared with ``ICToneDataset``:
        * no ``source_dir`` — images come from ``image_names.txt`` in the
          merged pair directory (absolute paths).
        * no ``lut_list_path`` — style/GT LUTs come from the pair; the content
          chain LUT pool defaults to the same ``lut_index.tsv``.
    """

    def __init__(
        self,
        pairs_npz: str | os.PathLike = "data/unified_A/sampled_merged.npz",
        image_names_txt: str | os.PathLike = "data/unified_A/image_names.txt",
        lut_index_tsv: str | os.PathLike = "data/unified_A/lut_index.tsv",
        content_lut_list_path: str | os.PathLike | None = None,
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        instance_prompt: str | None = None,
        length: int | None = None,
        repeats: int = 1,
        content_k_min: int = 2,
        content_k_max: int = 4,
        cache_luts: bool = True,
    ) -> None:
        if int(condition_size) != int(target_size):
            raise ValueError(
                "ICTonePairDataset requires condition_size == target_size; got "
                f"{condition_size} vs {target_size}."
            )

        # --- pairs ---
        d = np.load(pairs_npz)
        needed = ("q_img_idx", "r_img_idx", "q_lut_id", "r_lut_id")
        missing = [k for k in needed if k not in d.files]
        if missing:
            raise ValueError(
                f"pairs_npz {pairs_npz} is missing required columns: {missing}"
            )
        self.q_img_idx = np.asarray(d["q_img_idx"], dtype=np.int64)
        self.r_img_idx = np.asarray(d["r_img_idx"], dtype=np.int64)
        self.q_lut_id = np.asarray(d["q_lut_id"], dtype=np.int64)
        self.r_lut_id = np.asarray(d["r_lut_id"], dtype=np.int64)
        self.n_pairs = int(len(self.q_img_idx))
        if self.n_pairs == 0:
            raise RuntimeError(f"empty pairs npz: {pairs_npz}")

        # --- image_names.txt (absolute paths) ---
        self.image_paths: list[str] = Path(image_names_txt).read_text().splitlines()
        if not self.image_paths:
            raise RuntimeError(f"empty image_names.txt: {image_names_txt}")

        # --- lut_index.tsv -> {id: path} ---
        self.lut_by_id: dict[int, str] = {}
        for ln in Path(lut_index_tsv).read_text().splitlines()[1:]:
            if not ln.strip():
                continue
            idx_s, pth = ln.split("\t", 1)
            self.lut_by_id[int(idx_s)] = pth
        if not self.lut_by_id:
            raise RuntimeError(f"empty lut_index.tsv: {lut_index_tsv}")

        # --- content chain LUT pool (defaults to merged LUT set) ---
        if content_lut_list_path is not None:
            self.content_lut_paths: list[str] = _load_lut_list(content_lut_list_path)
        else:
            self.content_lut_paths = list(self.lut_by_id.values())
        if len(self.content_lut_paths) < content_k_max:
            raise RuntimeError(
                f"content LUT pool has {len(self.content_lut_paths)} entries, "
                f"need at least content_k_max={content_k_max}."
            )

        self.image_size = int(condition_size)
        self.drop_text_prob = float(drop_text_prob)
        self.instance_prompt = (
            instance_prompt if instance_prompt is not None else DEFAULT_INSTANCE_PROMPT
        )
        self.repeats = max(1, int(repeats))
        self._length = (
            int(length) if length is not None else self.n_pairs * self.repeats
        )
        self.content_k_min = int(content_k_min)
        self.content_k_max = int(content_k_max)
        self.to_tensor = T.ToTensor()

        self._cache_luts = bool(cache_luts)
        self._lut_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return self._length

    # ------------------------------------------------------------------
    def _load_lut_by_path(self, path: str):
        if self._cache_luts:
            lut = self._lut_cache.get(path)
            if lut is not None:
                return lut
        lut = load_cube_file(path)
        if self._cache_luts:
            self._lut_cache[path] = lut
        return lut

    def _sample_content_luts(self, k: int, exclude_paths: set[str]):
        """Sample ``k`` distinct LUTs (by path) from ``content_lut_paths``,
        avoiding paths in ``exclude_paths`` when possible.
        """
        luts: list = []
        paths: list[str] = []
        tried: set[int] = set()
        max_attempts = max(16, k * 8)
        attempts = 0
        pool = self.content_lut_paths
        while len(luts) < k and attempts < max_attempts:
            attempts += 1
            idx = random.randrange(len(pool))
            if idx in tried:
                continue
            tried.add(idx)
            p = pool[idx]
            if p in exclude_paths and len(pool) - len(exclude_paths) >= k - len(luts):
                continue
            if p in paths:
                continue
            try:
                lut = self._load_lut_by_path(p)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"content LUT load failed for {p}: {e}")
                continue
            luts.append(lut)
            paths.append(p)
        if len(luts) < k:
            raise RuntimeError(f"could not sample {k} content LUTs after {max_attempts} attempts")
        return luts, paths

    def _resolve_pair(self, i: int):
        """Return (q_img_path, r_img_path, q_lut_path, r_lut_path) for row i."""
        q_img_path = self.image_paths[int(self.q_img_idx[i])]
        r_img_path = self.image_paths[int(self.r_img_idx[i])]
        q_lut_path = self.lut_by_id[int(self.q_lut_id[i])]
        r_lut_path = self.lut_by_id[int(self.r_lut_id[i])]
        return q_img_path, r_img_path, q_lut_path, r_lut_path

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        # Try up to 8 pairs before giving up (handles bad images / broken LUTs).
        last_err: Exception | None = None
        for attempt in range(8):
            pair_idx = (index + attempt) % self.n_pairs if attempt == 0 \
                else random.randrange(self.n_pairs)
            try:
                q_img_path, r_img_path, q_lut_path, r_lut_path = \
                    self._resolve_pair(pair_idx)

                pil_a_raw = Image.open(q_img_path).convert("RGB")
                pil_b_raw = Image.open(r_img_path).convert("RGB")

                lut_q = self._load_lut_by_path(q_lut_path)   # style + GT LUT for q
                lut_r = self._load_lut_by_path(r_lut_path)   # style LUT for r
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.debug(f"pair {pair_idx} failed: {e}")
                continue
        else:
            raise RuntimeError(
                f"failed to prepare any pair after 8 attempts: {last_err}"
            )

        S = self.image_size
        pil_a = _resize_width_keep_aspect(pil_a_raw, S, height_multiple=16)
        H = pil_a.size[1]
        # Reference is force-resized to (S, H); see ICToneDataset comment.
        pil_b = pil_b_raw.convert("RGB").resize((S, H), Image.LANCZOS)

        def _apply(im: Image.Image, lut) -> Image.Image:
            try:
                out = im.filter(lut)
                if out.mode != "RGB":
                    out = out.convert("RGB")
                return out
            except Exception as e:  # noqa: BLE001
                logger.warning(f"LUT apply failed ({e}); using identity")
                return im.copy()

        def _apply_chain(im: Image.Image, lut_chain) -> Image.Image:
            out = im
            for lut in lut_chain:
                out = _apply(out, lut)
            return out

        # Content chain: K-1 random LUTs (K in [k_min, k_max]) applied to pil_a.
        # Avoid reusing the GT LUT so content != GT trivially.
        k = random.randint(self.content_k_min, self.content_k_max)
        content_luts, _paths = self._sample_content_luts(
            k=max(1, k - 1),
            exclude_paths={q_lut_path},
        )

        content = _apply_chain(pil_a, content_luts)
        style = _apply(pil_b, lut_r)     # r_image + r_lut
        gt = _apply(pil_a, lut_q)        # q_image + q_lut

        target = Image.new("RGB", (S * 3, H))
        target.paste(content, (0, 0))
        target.paste(style, (S, 0))
        target.paste(gt, (S * 2, 0))

        mask = Image.new("L", (S * 3, H), 0)
        ImageDraw.Draw(mask).rectangle([S * 2, 0, S * 3, H], fill=255)

        if random.random() < self.drop_text_prob:
            description = " "
        else:
            description = self.instance_prompt

        return {
            "image": self.to_tensor(target),
            "condition": self.to_tensor(mask),
            "condition_type": "ictone",
            "description": description,
            "position_delta": np.array([0, 0]),
        }


class ICToneTripletDataset(Dataset):
    """Triplet-driven dataset backed by a ``triplet.json`` file.

    Reads pre-built (content, reference, gt) image triples from a JSON list of
    dicts::

        [
            {"content": "content_images/00/00098.png",
             "reference": "style_images/000/00348_0049.png",
             "gt":        "style_images/000/00098_0049.png"},
            ...
        ]

    All three paths are relative to ``data_root`` (e.g. ``data/TST100K``). The
    ``reference`` and ``gt`` panels are used verbatim from the triplet — no
    LUT is applied to them. The content branch keeps the *content-degradation*
    pipeline: a chain of K random LUTs (K in ``[content_k_min, content_k_max]``)
    is applied to ``content`` before it is pasted into the left panel, so the
    model still has to learn to invert an arbitrary chain of colour edits
    while transferring the tone of ``reference``.

    Triptych = ``[ chain(content) | reference | gt ]``, mask over right third.

    Args:
        triplet_json: path to the triplet JSON file (list of dicts).
        data_root: root directory prepended to every relative path in the
            JSON. Set to ``""`` if paths in the JSON are already absolute.
        content_lut_list_path: text file with one ``.cube`` path per line —
            LUT pool for the content-degradation chain. Mutually exclusive
            with ``lut_index_tsv``. Defaults to ``None``; when both this and
            ``lut_index_tsv`` are ``None`` the content-degradation chain is
            disabled entirely and the content panel is used verbatim.
        lut_index_tsv: TSV file (``id\\tpath`` per row, header skipped) whose
            values seed the content LUT pool when ``content_lut_list_path`` is
            not provided. Defaults to ``None`` (no fallback pool).
        condition_size / target_size: per-part square size. Must match.
        drop_text_prob: probability of replacing the prompt with ``" "``.
        instance_prompt: text prompt for the triptych task. ``None`` →
            :data:`DEFAULT_INSTANCE_PROMPT`.
        length: virtual epoch length. ``None`` → ``|triplets| * repeats``.
        repeats: virtual epoch multiplier.
        content_k_min / content_k_max: content chain length is
            ``K ∈ [content_k_min, content_k_max]`` LUTs applied in order.
            ``K=0`` disables the chain entirely (identity content).
        cache_luts: cache each parsed .cube LUT in memory.

    Content is width-resized to ``condition_size`` keeping its aspect ratio
    (height rounded to a multiple of 16 for the VAE); reference and gt are
    force-resized to the same ``(S, H)``. The panel canvas is ``3S x H``;
    because H varies per sample, ``batch_size`` must be 1.
    """

    def __init__(
        self,
        triplet_json: str | os.PathLike = "data/TST100K/triplet.json",
        data_root: str | os.PathLike = "data/TST100K",
        content_lut_list_path: str | os.PathLike | None = None,
        lut_index_tsv: str | os.PathLike | None = None,
        condition_size: int = 512,
        target_size: int = 512,
        drop_text_prob: float = 0.1,
        instance_prompt: str | None = None,
        length: int | None = None,
        repeats: int = 1,
        content_k_min: int = 2,
        content_k_max: int = 4,
        cache_luts: bool = True,
    ) -> None:
        if int(condition_size) != int(target_size):
            raise ValueError(
                "ICToneTripletDataset requires condition_size == target_size; "
                f"got {condition_size} vs {target_size}."
            )

        # --- triplets ---
        with open(triplet_json, "r") as f:
            raw = json.load(f)
        if not isinstance(raw, list) or not raw:
            raise RuntimeError(f"empty or malformed triplet json: {triplet_json}")
        needed = {"content", "reference", "gt"}
        missing = [k for k in needed if k not in raw[0]]
        if missing:
            raise ValueError(
                f"triplet json {triplet_json} missing required keys {missing} "
                f"in the first entry: {raw[0]!r}"
            )
        self.data_root = str(data_root) if data_root else ""
        self.triplets: list[dict] = raw
        self.n_triplets = len(self.triplets)

        # --- content-chain LUT pool (optional) ---
        # ``content_lut_list_path is None`` and no ``lut_index_tsv`` -> the
        # content-degradation chain is disabled and the content panel is
        # pasted verbatim.
        if content_lut_list_path is not None:
            self.content_lut_paths: list[str] = _load_lut_list(content_lut_list_path)
        elif lut_index_tsv is not None:
            paths: list[str] = []
            for ln in Path(lut_index_tsv).read_text().splitlines()[1:]:
                if not ln.strip():
                    continue
                _idx_s, pth = ln.split("\t", 1)
                paths.append(pth)
            if not paths:
                raise RuntimeError(f"empty lut_index.tsv: {lut_index_tsv}")
            self.content_lut_paths = paths
        else:
            self.content_lut_paths = []

        self._content_deg_enabled = len(self.content_lut_paths) > 0
        if self._content_deg_enabled and len(self.content_lut_paths) < max(1, content_k_max):
            raise RuntimeError(
                f"content LUT pool has {len(self.content_lut_paths)} entries, "
                f"need at least content_k_max={content_k_max}."
            )

        self.image_size = int(condition_size)
        self.drop_text_prob = float(drop_text_prob)
        self.instance_prompt = (
            instance_prompt if instance_prompt is not None else DEFAULT_INSTANCE_PROMPT
        )
        self.repeats = max(1, int(repeats))
        self._length = (
            int(length) if length is not None else self.n_triplets * self.repeats
        )
        self.content_k_min = int(content_k_min)
        self.content_k_max = int(content_k_max)
        self.to_tensor = T.ToTensor()

        self._cache_luts = bool(cache_luts)
        self._lut_cache: dict[str, object] = {}

    def __len__(self) -> int:
        return self._length

    # ------------------------------------------------------------------
    def _resolve(self, rel_path: str) -> str:
        return os.path.join(self.data_root, rel_path) if self.data_root else rel_path

    def _load_lut_by_path(self, path: str):
        if self._cache_luts:
            lut = self._lut_cache.get(path)
            if lut is not None:
                return lut
        lut = load_cube_file(path)
        if self._cache_luts:
            self._lut_cache[path] = lut
        return lut

    def _sample_content_luts(self, k: int):
        """Sample ``k`` distinct LUTs (by path) from ``content_lut_paths``."""
        if k <= 0:
            return [], []
        luts: list = []
        paths: list[str] = []
        tried: set[int] = set()
        max_attempts = max(16, k * 8)
        attempts = 0
        pool = self.content_lut_paths
        while len(luts) < k and attempts < max_attempts:
            attempts += 1
            idx = random.randrange(len(pool))
            if idx in tried:
                continue
            tried.add(idx)
            p = pool[idx]
            if p in paths:
                continue
            try:
                lut = self._load_lut_by_path(p)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"content LUT load failed for {p}: {e}")
                continue
            luts.append(lut)
            paths.append(p)
        if len(luts) < k:
            raise RuntimeError(
                f"could not sample {k} content LUTs after {max_attempts} attempts"
            )
        return luts, paths

    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> dict:
        # Try up to 8 rows before giving up (handles bad images / broken LUTs).
        last_err: Exception | None = None
        for attempt in range(8):
            row_idx = (index + attempt) % self.n_triplets if attempt == 0 \
                else random.randrange(self.n_triplets)
            row = self.triplets[row_idx]
            try:
                content_pil = Image.open(self._resolve(row["content"])).convert("RGB")
                ref_pil = Image.open(self._resolve(row["reference"])).convert("RGB")
                gt_pil = Image.open(self._resolve(row["gt"])).convert("RGB")
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.debug(f"triplet {row_idx} failed: {e}")
                continue
        else:
            raise RuntimeError(
                f"failed to prepare any triplet after 8 attempts: {last_err}"
            )

        S = self.image_size
        content_pil_r = _resize_width_keep_aspect(content_pil, S, height_multiple=16)
        H = content_pil_r.size[1]
        ref_pil_r = ref_pil.convert("RGB").resize((S, H), Image.LANCZOS)
        gt_pil_r = gt_pil.convert("RGB").resize((S, H), Image.LANCZOS)

        def _apply(im: Image.Image, lut) -> Image.Image:
            try:
                out = im.filter(lut)
                if out.mode != "RGB":
                    out = out.convert("RGB")
                return out
            except Exception as e:  # noqa: BLE001
                logger.warning(f"LUT apply failed ({e}); using identity")
                return im.copy()

        def _apply_chain(im: Image.Image, lut_chain) -> Image.Image:
            out = im
            for lut in lut_chain:
                out = _apply(out, lut)
            return out

        # Content-degradation chain: K random LUTs (K in [k_min, k_max]).
        # K=0 → no LUT applied (identity content). When the LUT pool is empty
        # (``content_lut_list_path`` / ``lut_index_tsv`` both None) the chain
        # is skipped entirely and the content panel is used verbatim.
        if self._content_deg_enabled:
            k = random.randint(self.content_k_min, self.content_k_max)
            content_luts, _paths = self._sample_content_luts(k=max(0, k))
            content_deg = _apply_chain(content_pil_r, content_luts)
        else:
            content_deg = content_pil_r

        # Triptych = [degraded content | reference | gt].
        target = Image.new("RGB", (S * 3, H))
        target.paste(content_deg, (0, 0))
        target.paste(ref_pil_r, (S, 0))
        target.paste(gt_pil_r, (S * 2, 0))

        mask = Image.new("L", (S * 3, H), 0)
        ImageDraw.Draw(mask).rectangle([S * 2, 0, S * 3, H], fill=255)

        if random.random() < self.drop_text_prob:
            description = " "
        else:
            description = self.instance_prompt

        return {
            "image": self.to_tensor(target),
            "condition": self.to_tensor(mask),
            "condition_type": "ictone",
            "description": description,
            "position_delta": np.array([0, 0]),
        }
