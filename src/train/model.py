import lightning as L
from diffusers.pipelines import FluxPipeline, FluxFillPipeline
from diffusers.utils import convert_unet_state_dict_to_peft
import torch
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
import os
import safetensors.torch
import prodigyopt

from ..flux.pipeline_tools import encode_images_fill, prepare_text_input


class OminiModel(L.LightningModule):
    def __init__(
        self,
        flux_fill_id: str,
        lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config

        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_fill_pipe = FluxFillPipeline.from_pretrained(flux_fill_id).to(dtype=dtype).to(device)

        self.transformer = self.flux_fill_pipe.transformer
        self.text_encoder = self.flux_fill_pipe.text_encoder
        self.text_encoder_2 = self.flux_fill_pipe.text_encoder_2
        # Older diffusers accepted a bare `.gradient_checkpointing = True`
        # attribute, but current versions require `enable_gradient_checkpointing()`
        # (which also installs `_gradient_checkpointing_func`, otherwise the
        # transformer forward crashes with `'NoneType' object is not callable`).
        if gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()
        else:
            self.transformer.gradient_checkpointing = False
        self.transformer.train()
        # Freeze the Flux pipeline
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        self.flux_fill_pipe.vae.requires_grad_(False).eval()
        self.use_offset_noise = use_offset_noise

        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')

        self.lora_layers = self.init_lora(lora_path, lora_config)

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        """Attach a LoRA adapter, optionally warm-starting weights from ``lora_path``.

        ``lora_config`` is always required (defines rank / alpha / target
        modules / init scheme). ``lora_path`` is optional; when provided,
        weights are loaded from either a ``pytorch_lora_weights.safetensors``
        file directly or a directory containing one, and copied into the
        freshly-attached PEFT adapter under the ``default`` name.

        The rank and target-module regex in ``lora_config`` must match the
        checkpoint or ``set_peft_model_state_dict`` will complain / drop keys.
        """
        assert lora_config, "lora_config is required to build the LoRA adapter"
        self.transformer.add_adapter(LoraConfig(**lora_config))
        lora_layers = list(
            filter(lambda p: p.requires_grad, self.transformer.parameters())
        )

        if lora_path:
            if os.path.isfile(lora_path):
                lora_sd = safetensors.torch.load_file(lora_path, device="cpu")
            else:
                lora_sd = FluxFillPipeline.lora_state_dict(lora_path)
            transformer_sd = {
                k.replace("transformer.", ""): v
                for k, v in lora_sd.items()
                if k.startswith("transformer.")
            }
            transformer_sd = convert_unet_state_dict_to_peft(transformer_sd)
            incompat = set_peft_model_state_dict(
                self.transformer, transformer_sd, adapter_name="default",
            )
            print(f"[lora_init] loaded {len(transformer_sd)} tensors from {lora_path}")
            if incompat is not None:
                miss = getattr(incompat, "missing_keys", None) or []
                unexp = getattr(incompat, "unexpected_keys", None) or []
                if miss or unexp:
                    print(
                        f"[lora_init] incompatible keys: "
                        f"missing={len(miss)}, unexpected={len(unexp)}"
                    )
                    if unexp[:3]:
                        print(f"[lora_init]   e.g. unexpected: {unexp[:3]}")
                    if miss[:3]:
                        print(f"[lora_init]   e.g. missing:    {miss[:3]}")

        return lora_layers

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )
        if self.model_config['use_sep']:
            torch.save(self.text_encoder_2.shared, os.path.join(path, "t5_embedding.pth"))
            torch.save(self.text_encoder.text_model.embeddings.token_embedding, os.path.join(path, "clip_embedding.pth"))

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        mask_imgs = batch["condition"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]

        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_fill_pipe, prompts
            )

            x_0, x_cond, img_ids = encode_images_fill(self.flux_fill_pipe, imgs, mask_imgs, prompt_embeds.dtype, prompt_embeds.device)

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)

            if self.use_offset_noise:
                x_1 = x_1 + 0.1 * torch.randn(x_1.shape[0], 1, x_1.shape[2]).to(self.device).to(self.dtype)

            t_ = t.unsqueeze(1).unsqueeze(1)
            x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = self.transformer(
            hidden_states=torch.cat((x_t, x_cond), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        self.last_t = t.mean().item()
        return loss
