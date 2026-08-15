# Checkpoints

Three sets of checkpoints are needed for ICTone inference / evaluation. Download the ones you need — commands below are meant to be **copy-pasted individually**, not run in one shot.

> All commands should be executed **from the `ICTone/` directory**. Optional: `export HF_HUB_ENABLE_HF_TRANSFER=1` to speed up large-file downloads.

## Overview

| Name | Repo | Used for | Access |
|---|---|---|---|
| `metric_weights` | [ToneStyle/MetricCkpt](https://huggingface.co/ToneStyle/MetricCkpt) | Evaluation (AesCLIP / CDFlow / LDC / CLIP) | Public |
| `FLUX-Fill` | [black-forest-labs/FLUX.1-Fill-dev](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev) | Base model for training & inference | **Gated — HF token required** |
| `lora_2500` | [ToneStyle/FLUX-Fill-Lora](https://huggingface.co/ToneStyle/FLUX-Fill-Lora) | ICTone LoRA weights (inference) | Public |

---

## 1) metric_weights (required only for evaluation)

```bash
huggingface-cli download ToneStyle/MetricCkpt \
    --local-dir ckpt/metric_weights \
    --local-dir-use-symlinks False
```

## 2) FLUX-Fill-dev (base model, gated)

Before running, request access on the model page and generate a token:
- Request access: <https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev>
- Create a token: <https://huggingface.co/settings/tokens>

```bash
huggingface-cli download black-forest-labs/FLUX.1-Fill-dev \
    --local-dir ckpt/FLUX-Fill \
    --local-dir-use-symlinks False \
    --token hf_xxxxxxxxxxxxxxxxxxxx
```

## 3) ICTone LoRA (inference weights)

```bash
huggingface-cli download ToneStyle/ICTone-Fill-LoRA \
    --local-dir ckpt/ICTone-Fill-LoRA \
    --local-dir-use-symlinks False
```
