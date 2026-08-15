import lightning as L
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
from transformers import pipeline
# import cv2
import torch
import os
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")
        # Dump the raw training batch (triptych image + mask) every N steps to
        # runs/<run>/train_inputs/step<N>/ for eyeball sanity-checking. Set to
        # 0 or None to disable. Only the first item of the batch is written.
        tv = training_config.get("train_input_vis_interval", 0)
        self.train_input_vis_interval = int(tv) if tv else 0
        self.train_input_vis_max = int(
            training_config.get("train_input_vis_max", 2)
        )
        # Retain only the most recent K LoRA checkpoints. None / <=0 disables pruning.
        keep_last_k = training_config.get("keep_last_k", None)
        self.keep_last_k = int(keep_last_k) if keep_last_k is not None else None

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )

        self.total_steps = 0

    def on_train_start(self, trainer, pl_module):
        """Run one validation pass BEFORE any optimizer step (step-0 sample).

        Only fires on the main process to avoid duplicated work on multi-GPU
        (Lightning attaches the callback to every rank). The pipeline used
        inside ``_generate_ictone_samples`` is the shared ``pl_module.flux_fill_pipe``,
        so we can invoke it directly without rebuilding.
        """
        if not getattr(trainer, "is_global_zero", True):
            return

        cond_type = self.training_config.get("condition_type", "ictone")
        # Per-step validation subdir: runs/<run_name>/validate/step<N>/
        val_save_path = os.path.join(
            self.save_path, self.run_name, "validate", f"step{self.total_steps}"
        )
        os.makedirs(val_save_path, exist_ok=True)

        print(
            f"Epoch: 0, Steps: 0 - Running pre-training validation "
            f"(condition_type={cond_type})"
        )
        try:
            self.generate_a_sample(
                trainer,
                pl_module,
                val_save_path,
                f"lora_{self.total_steps}",
                cond_type,
            )
        except Exception as e:  # noqa: BLE001
            # Non-fatal: log and continue with training. A failing initial
            # validation shouldn't take down a long training run.
            print(f"[warn] step-0 validation failed: {e}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Dump the raw training batch as an image for visual sanity checks.
        # Runs on the main process only. Wrapped in try/except so a broken
        # sample never kills training.
        if (
            self.train_input_vis_interval > 0
            and getattr(trainer, "is_global_zero", True)
            and self.total_steps % self.train_input_vis_interval == 0
        ):
            try:
                self._dump_train_inputs(batch)
            except Exception as e:  # noqa: BLE001
                print(f"[train-vis] failed to dump inputs at step {self.total_steps}: {e}")

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)

        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )
            self._prune_old_ckpts()

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            val_save_path = os.path.join(
                self.save_path, self.run_name, "validate", f"step{self.total_steps}"
            )
            os.makedirs(val_save_path, exist_ok=True)
            self.generate_a_sample(
                trainer,
                pl_module,
                val_save_path,
                f"lora_{self.total_steps}",
                batch["condition_type"][
                    0
                ],  # Use the condition type from the current batch
            )

    def _dump_train_inputs(self, batch) -> None:
        """Save the current training batch as PNGs for visual inspection.

        Writes to ``{save_path}/{run_name}/train_inputs/step{N}/``:
          - ``sample{i}_input.png``  : the triptych fed to the model (image).
          - ``sample{i}_mask.png``   : the inpaint mask (right third white).
          - ``sample{i}_masked.png`` : image * (1 - mask), i.e. what the
            model actually sees on the RHS (right panel blanked).
          - ``sample{i}_prompt.txt`` : the text prompt (or a placeholder).

        Batch tensors follow the ``TST2KTripletDataset`` convention: images in
        [0, 1], shape (B, 3, H, 3S); mask in [0, 1], shape (B, 1, H, 3S).
        """
        img = batch.get("image")
        cond = batch.get("condition")
        if img is None or cond is None:
            return

        out_dir = os.path.join(
            self.save_path, self.run_name, "train_inputs", f"step{self.total_steps}"
        )
        os.makedirs(out_dir, exist_ok=True)

        # Detach + move to CPU once. Handle both (B, C, H, W) and (C, H, W).
        img_cpu = img.detach().float().cpu()
        cond_cpu = cond.detach().float().cpu()
        if img_cpu.ndim == 3:
            img_cpu = img_cpu.unsqueeze(0)
            cond_cpu = cond_cpu.unsqueeze(0)

        b = min(img_cpu.shape[0], max(1, self.train_input_vis_max))
        descs = batch.get("description")
        if isinstance(descs, str):
            descs = [descs]
        elif descs is None:
            descs = [""] * b

        def _to_pil_rgb(t: torch.Tensor) -> Image.Image:
            arr = (t.clamp(0, 1) * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
            return Image.fromarray(arr, mode="RGB")

        def _to_pil_l(t: torch.Tensor) -> Image.Image:
            arr = (t.clamp(0, 1) * 255.0).round().to(torch.uint8).squeeze(0).numpy()
            return Image.fromarray(arr, mode="L")

        for i in range(b):
            _to_pil_rgb(img_cpu[i]).save(os.path.join(out_dir, f"sample{i}_input.png"))
            _to_pil_l(cond_cpu[i]).save(os.path.join(out_dir, f"sample{i}_mask.png"))
            # masked_image = image * (1 - mask): mirrors OminiModel.step so
            # the dump matches what the transformer actually sees.
            masked = img_cpu[i] * (1.0 - cond_cpu[i])
            _to_pil_rgb(masked).save(os.path.join(out_dir, f"sample{i}_masked.png"))
            with open(os.path.join(out_dir, f"sample{i}_prompt.txt"), "w") as f:
                f.write(str(descs[i]) if i < len(descs) else "")

    def _prune_old_ckpts(self) -> None:
        """Keep only the ``keep_last_k`` most recent LoRA checkpoint dirs.

        Checkpoint dirs are named by ``self.total_steps`` under
        ``{save_path}/{run_name}/ckpt/``. Older ones (smaller step number) are
        removed. No-op when ``keep_last_k`` is None or non-positive.
        """
        if not self.keep_last_k or self.keep_last_k <= 0:
            return
        ckpt_root = os.path.join(self.save_path, self.run_name, "ckpt")
        if not os.path.isdir(ckpt_root):
            return
        entries = []
        for name in os.listdir(ckpt_root):
            full = os.path.join(ckpt_root, name)
            if not os.path.isdir(full):
                continue
            try:
                step = int(name)
            except ValueError:
                continue
            entries.append((step, full))
        entries.sort(key=lambda x: x[0])
        excess = len(entries) - self.keep_last_k
        if excess <= 0:
            return
        import shutil
        for _step, path in entries[:excess]:
            try:
                shutil.rmtree(path)
                print(f"[ckpt] pruned old LoRA checkpoint: {path}")
            except Exception as e:  # noqa: BLE001
                print(f"[ckpt] failed to remove {path}: {e}")

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type,
    ):
        # ICTone filter-migration validation. Iterates the first N TST2K subdirs
        # (each has content.png + reference.png), builds the same
        # ``[content | reference | content]`` triptych input as inference, and
        # runs the FluxFill pipeline. Saved panel: content | reference | pred (+ gt).
        if condition_type == "ictone":
            self._generate_ictone_samples(pl_module, save_path)
            return

    @torch.no_grad()
    def _generate_ictone_samples(self, pl_module, save_path):
        """Validation panel generator for the ICTone triptych task.

        Iterates the first ``ictone_val_num`` subdirs of
        ``ictone_val_dir`` (each must contain ``content.png`` +
        ``reference.png``). For each, builds a
        ``[content | reference | content]`` triptych (right = identity-prior
        for the fill region), runs the FluxFill pipeline once, and saves a
        horizontal panel ``content | reference | model_output [| gt]`` under
        ``save_path``.
        """
        from pathlib import Path
        cfg = self.training_config or {}
        val_dir = Path(cfg.get("ictone_val_dir", "data/TST2K"))
        val_num = int(cfg.get("ictone_val_num", 8))
        val_steps = int(cfg.get("ictone_val_inference_steps", 28))
        val_guidance = float(cfg.get("ictone_val_guidance_scale", 30.0))
        val_prompt = cfg.get(
            "ictone_val_prompt",
            (
                "A side-by-side triptych. Left: source photo. "
                "Middle: a color and tone reference photo. "
                "Right: the same scene as the left, re-graded so its colors, "
                "contrast, and film look match the middle reference, while "
                "preserving the left's content and details."
            ),
        )
        ds_cfg = cfg.get("dataset", {}) or {}
        image_size = int(ds_cfg.get("condition_size", 512))
        seed_base = int(cfg.get("ictone_val_seed", 666))

        subs = sorted([p for p in val_dir.iterdir() if p.is_dir()])
        items = []
        for sub in subs:
            c = sub / "content.png"
            r = sub / "reference.png"
            if not (c.exists() and r.exists()):
                continue
            items.append(sub)
            if len(items) >= val_num:
                break
        if not items:
            print(f"[ictone-val] no valid items under {val_dir}")
            return

        S = image_size

        def _resize_to_wh(p, width: int, height: int) -> Image.Image:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            s = max(width / w, height / h)
            nw = max(int(round(w * s)), width)
            nh = max(int(round(h * s)), height)
            im = im.resize((nw, nh), Image.LANCZOS)
            left = (nw - width) // 2
            top = (nh - height) // 2
            return im.crop((left, top, left + width, top + height))

        def _resize_width_keep_aspect(p, width: int, height_multiple: int = 16) -> Image.Image:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            new_h = int(round(h * width / w))
            new_h = max(height_multiple, int(round(new_h / height_multiple)) * height_multiple)
            return im.resize((width, new_h), Image.LANCZOS)

        # Keep the pipeline entirely in its native dtype (bf16 per OminiModel).
        # Casting only the VAE to fp32 breaks conv2d inside VAE encode/decode
        # ("Input type BFloat16 and bias type float should be the same") because
        # the pipeline still feeds bf16 tensors. The base FluxFill VAE runs
        # fine at bf16 for inference (that's how ICEdit itself uses it).
        pl_module.flux_fill_pipe.transformer.eval()
        try:
            for i, sub in enumerate(items):
                # Validation always uses aspect-preserving resize: width -> S,
                # height keeps content's aspect ratio (rounded to /16 for the
                # VAE). Reference is force-resized (no crop, no aspect keep)
                # to the same (S, H) so all three triptych panels align.
                content_r = _resize_width_keep_aspect(sub / "content.png", S)
                H = content_r.size[1]
                ref_r = Image.open(sub / "reference.png").convert("RGB").resize(
                    (S, H), Image.LANCZOS
                )

                canvas_w = 3 * S
                canvas_h = H
                input_img = Image.new("RGB", (canvas_w, canvas_h))
                input_img.paste(content_r, (0, 0))
                input_img.paste(ref_r, (S, 0))
                input_img.paste(content_r, (2 * S, 0))  # identity prior

                mask = Image.new("L", (canvas_w, canvas_h), 0)
                draw = ImageDraw.Draw(mask)
                draw.rectangle([2 * S, 0, 3 * S, canvas_h], fill=255)

                device = pl_module.flux_fill_pipe._execution_device
                out = pl_module.flux_fill_pipe(
                    prompt=val_prompt,
                    image=input_img,
                    height=canvas_h,
                    width=canvas_w,
                    mask_image=mask,
                    guidance_scale=val_guidance,
                    num_inference_steps=val_steps,
                    max_sequence_length=512,
                    generator=torch.Generator(device=device).manual_seed(seed_base + i),
                ).images[0]

                # Save the [content | reference | model_output (| gt)] panel.
                pred_right = out.crop((2 * S, 0, 3 * S, canvas_h))
                tiles = [content_r, ref_r, pred_right]
                gt_path = sub / "gt.png"
                if gt_path.exists():
                    tiles.append(_resize_to_wh(gt_path, S, canvas_h))
                panel = Image.new("RGB", (S * len(tiles), canvas_h))
                for j, t in enumerate(tiles):
                    panel.paste(t, (S * j, 0))
                panel.save(
                    os.path.join(save_path, f"ictone-step{self.total_steps}-{i:03d}-{sub.name}.jpg")
                )
        finally:
            pl_module.flux_fill_pipe.transformer.train()
