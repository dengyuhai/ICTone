"""Hugging Face Spaces demo for ICTone tone/style transfer.

Run locally from this directory:
    python app.py

For a Space, set ``FLUX_PATH`` and ``LORA_PATH`` as environment variables or
replace their defaults below. ``FLUX_PATH`` may be a local directory or a
Hugging Face model id.
"""
from __future__ import annotations

import os
import random
from functools import lru_cache

import gradio as gr
import numpy as np
import torch
from diffusers import FluxFillPipeline
from PIL import Image

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(function):
            return function

    spaces = _SpacesFallback()

from inference import (
    DEFAULT_INSTANCE_PROMPT,
    apply_lut,
    estimate_lut,
    run_one,
)

MAX_SEED = np.iinfo(np.int32).max
DEFAULT_FLUX_PATH = os.getenv(
    "FLUX_PATH", "black-forest-labs/FLUX.1-Fill-dev"
)
DEFAULT_LORA_PATH = os.getenv("LORA_PATH", "ToneStyle/FLUX-Fill-Lora")
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "512"))


def _load_pipeline() -> FluxFillPipeline:
    if not torch.cuda.is_available():
        raise RuntimeError("ICTone requires a CUDA GPU to run this demo.")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    pipe = FluxFillPipeline.from_pretrained(
        DEFAULT_FLUX_PATH,
        torch_dtype=dtype,
    )
    if DEFAULT_LORA_PATH:
        pipe.load_lora_weights(DEFAULT_LORA_PATH)

    if os.getenv("ENABLE_MODEL_CPU_OFFLOAD", "1").lower() in {"1", "true", "yes"}:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


@lru_cache(maxsize=1)
def get_pipeline() -> FluxFillPipeline:
    return _load_pipeline()


@spaces.GPU
def infer(
    content: Image.Image,
    reference: Image.Image,
    seed: int,
    randomize_seed: bool,
    guidance_scale: float,
    num_inference_steps: int,
    lut_size: int,
    progress=gr.Progress(track_tqdm=True),
):
    if content is None or reference is None:
        raise gr.Error("Please upload both a content image and a reference image.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    pred, panel, _, _ = run_one(
        get_pipeline(),
        content.convert("RGB"),
        reference.convert("RGB"),
        size=IMAGE_SIZE,
        prompt=DEFAULT_INSTANCE_PROMPT,
        guidance_scale=float(guidance_scale),
        num_inference_steps=int(num_inference_steps),
        seed=seed,
        generator_device="cuda",
    )

    # Reconstruct at the original content resolution, matching inference.py.
    if lut_size > 1:
        content_rgb = content.convert("RGB")
        before = np.asarray(content_rgb)
        after = np.asarray(pred.resize(content_rgb.size, Image.BILINEAR))
        before_flat = before.reshape(-1, 3)
        after_flat = after.reshape(-1, 3)
        max_samples = 500_000
        if len(before_flat) > max_samples:
            rng = np.random.default_rng(0)
            selected = rng.choice(len(before_flat), max_samples, replace=False)
            before_flat = before_flat[selected]
            after_flat = after_flat[selected]
        lut = estimate_lut(
            before_flat,
            after_flat,
            size=int(lut_size),
            device="cuda" if torch.cuda.is_available() else None,
        )
        output = Image.fromarray(apply_lut(before, lut, device="cuda"))
    else:
        output = pred

    return output, panel, seed


with gr.Blocks(title="ICTone Tone Transfer") as demo:
    gr.Markdown("# ICTone Tone Transfer")
    gr.Markdown(
        "Upload a content image and a reference image. ICTone transfers the "
        "reference color, contrast, and film look while preserving content."
    )

    with gr.Row():
        content = gr.Image(label="Content image", type="pil")
        reference = gr.Image(label="Reference image", type="pil")

    with gr.Row():
        seed = gr.Number(label="Seed", value=666, precision=0)
        randomize_seed = gr.Checkbox(label="Randomize seed", value=False)
        guidance = gr.Slider(
            label="Guidance scale", minimum=1, maximum=100, value=50, step=1
        )
        steps = gr.Slider(
            label="Inference steps", minimum=1, maximum=28, value=4, step=1
        )
        lut = gr.Slider(
            label="LUT size", minimum=0, maximum=33, value=33, step=1
        )

    run = gr.Button("Transfer tone", variant="primary")
    with gr.Row():
        output = gr.Image(label="Result", type="pil")
        preview = gr.Image(label="Content | Reference | Result", type="pil")
    used_seed = gr.Number(label="Used seed", precision=0)

    run.click(
        infer,
        inputs=[content, reference, seed, randomize_seed, guidance, steps, lut],
        outputs=[output, preview, used_seed],
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name=os.getenv("SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("PORT", "7860")),
    )
