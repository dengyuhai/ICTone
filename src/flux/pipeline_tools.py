from diffusers.pipelines import FluxPipeline, FluxFillPipeline
from diffusers.utils import logging
from diffusers.pipelines.flux.pipeline_flux import logger
from torch import Tensor
import torch


def encode_images(pipeline: FluxPipeline, images: Tensor):
    images = pipeline.image_processor.preprocess(images)
    images = images.to(pipeline.device).to(pipeline.dtype)
    images = pipeline.vae.encode(images).latent_dist.sample()
    images = (
        images - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    images_tokens = pipeline._pack_latents(images, *images.shape)
    images_ids = pipeline._prepare_latent_image_ids(
        images.shape[0],
        images.shape[2],
        images.shape[3],
        pipeline.device,
        pipeline.dtype,
    )
    if images_tokens.shape[1] != images_ids.shape[0]:
        images_ids = pipeline._prepare_latent_image_ids(
            images.shape[0],
            images.shape[2] // 2,
            images.shape[3] // 2,
            pipeline.device,
            pipeline.dtype,
        )
    return images_tokens, images_ids



def encode_images_fill(pipeline: FluxFillPipeline, image: Tensor, mask_image: Tensor, dtype: torch.dtype, device: str):
    images_tokens, images_ids = encode_images(pipeline, image.clone().detach())
    height, width = image.shape[-2:]
    # print(f"height: {height}, width: {width}")
    image = pipeline.image_processor.preprocess(image, height=height, width=width)
    mask_image = pipeline.mask_processor.preprocess(mask_image, height=height, width=width)

    masked_image = image * (1 - mask_image)
    masked_image = masked_image.to(device=device, dtype=dtype)

    num_channels_latents = pipeline.vae.config.latent_channels
    height, width = image.shape[-2:]
    device = pipeline._execution_device
    mask, masked_image_latents = pipeline.prepare_mask_latents(
        mask_image,
        masked_image,
        image.shape[0],
        num_channels_latents,
        1,
        height,
        width,
        dtype,
        device,
        None,
    )
    masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1)
    return images_tokens, masked_image_latents, images_ids


def prepare_text_input(pipeline: FluxPipeline, prompts, max_sequence_length=512):
    # Turn off warnings (CLIP overflow)
    logger.setLevel(logging.ERROR)
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = pipeline.encode_prompt(
        prompt=prompts,
        prompt_2=None,
        prompt_embeds=None,
        pooled_prompt_embeds=None,
        device=pipeline.device,
        num_images_per_prompt=1,
        max_sequence_length=max_sequence_length,
        lora_scale=None,
    )
    # Turn on warnings
    logger.setLevel(logging.WARNING)
    return prompt_embeds, pooled_prompt_embeds, text_ids
