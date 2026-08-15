**English** | [简体中文](README_zh.md)

# [ECCV' 2026] Towards In-Context Tone Style Transfer with a Large-Scale Triplet Dataset

<!-- **Towards In-Context Tone Style Transfer with a Large-Scale Triplet Dataset** -->
<a href="https://arxiv.org/pdf/2604.16114"><img src="https://img.shields.io/badge/arXiv-2604.16114-b31b1b.svg" alt="Paper"></a>
<a href="https://dengyuhai.github.io/ICTone_Project/"><img src="https://img.shields.io/badge/Project%20Page-Visit-blue" alt="Project Page"></a>
<a href="https://huggingface.co/spaces/ToneStyle/ICTone-Fill"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HF%20Demo-yellow.svg" alt="Hugging Face Demo"></a>

<a href="https://huggingface.co/ToneStyle/ICTone-Fill-LoRA"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Model%20Weights-green.svg" alt="Model Weights"></a>
<a href="https://huggingface.co/datasets/ToneStyle/TST100K"><img src="https://img.shields.io/badge/%F0%9F%A4%97-TST100K%20Dataset-blue.svg" alt="TST100K Dataset"></a>
<a href="https://huggingface.co/datasets/ToneStyle/TST2K"><img src="https://img.shields.io/badge/%F0%9F%A4%97-TST2K%20Benchmark-blueviolet.svg" alt="TST2K Benchmark"></a>

> Yuhai Deng, Huimin She, Wei Shen, Meng Li, Ruoxi Wu, Lunxi Yuan, Xiang Li
>
> Nankai University · OPPO AI Center

## 📝 Overview

![ICTone teaser](assets/teaser.jpg)

**ICTone** formulates **reference-based tone style transfer** as an **in-context generation** task. Unlike conventional methods that use two separate encoders to extract content and reference features before fusing them in a decoder, ICTone provides the content and reference images to a diffusion transformer as a **joint context**. This design allows the model to leverage the semantic priors of generative foundation models for semantics-aware tone transfer, substantially reducing improper color transfer and semantic misalignment—for example, incorrectly transferring the warm color of a railing onto a person's face.

## 📮 News

- **[2026.08.15]** The ICTone online demo is now live on Hugging Face: [🤗 ICTone Demo](https://huggingface.co/spaces/ToneStyle/ICTone-Fill).
- **[2026.08.14]** The ICTone LoRA weights are now available for direct inference: [🤗 ToneStyle/ICTone-Fill-LoRA](https://huggingface.co/ToneStyle/ICTone-Fill-LoRA).
- **[2026.08.13]** Our large-scale triplet dataset, **TST100K** (100K+ content/reference/ground-truth triplets), is now available: [🤗 ToneStyle/TST100K](https://huggingface.co/datasets/ToneStyle/TST100K).
- **[2026.07.15]** The **TST2K** benchmark is now available: [🤗 ToneStyle/TST2K](https://huggingface.co/datasets/ToneStyle/TST2K). It covers portrait, food, landscape, night, and lifestyle scenes.
- **[2026.06.18]** ICTone was accepted to ECCV 2026.

## 💻 Environment

We recommend using a dedicated Conda environment. The project has been tested with **Python 3.10 and CUDA GPUs with BF16 support**.

```bash
# 1. Create and activate the environment
conda create -n ictone python=3.10 -y
conda activate ictone

# 2. Install dependencies
cd ICTone
pip install -r requirements.txt
```

## 🚀 Quick Inference

### 🖼️ Single Image Pair

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --content   assets/example_content.jpg \
    --reference assets/example_reference.jpg \
    --flux-path black-forest-labs/FLUX.1-Fill-dev \
    --lora-path ToneStyle/ICTone-Fill-LoRA \
    --output-file ./output/example_output.png \
    --num-inference-steps 4 \
    --guidance-scale 50.0 \
    --seed 666
```

## 🛠️ Training and Evaluation

### 🔧 Training

**Data:** Download [🤗 ToneStyle/TST100K](https://huggingface.co/datasets/ToneStyle/TST100K) to `data/TST100K/`. The dataset contains 100K+ triplets organized through `content_images/`, `style_images/`, and `triplet.json`. See [`data/DOWNLOAD.md`](data/DOWNLOAD.md) for detailed download instructions.

**Configuration:** `configs/ictone_pair_lora.yaml`

- `flux_path`: the FLUX-Fill model directory. The default is the relative path `ckpt/FLUX-Fill`; run training and inference from the `ICTone/` directory. A Hugging Face model repository ID is also supported.
- `train.dataset.triplet_json` / `data_root`: the triplet manifest and dataset root.
- `train.max_steps` / `save_interval` / `sample_interval` / `keep_last_k`: the maximum number of training steps, checkpoint interval, visualization interval, and number of recent checkpoints to retain. Our setup uses four A100 GPUs for 50K steps with `lr=1e-4` and `weight_decay=1e-3`.
- `train.ictone_val_*`: periodic visualization settings for validation on `data/TST2K` during training.

**Launch:**

```bash
# Single-GPU smoke test
GPUS=0 bash train_ictone.sh

# Default four-GPU run
bash train_ictone.sh
```

### 📊 Evaluation

**Data preparation**

Download the [🤗 ToneStyle/TST2K](https://huggingface.co/datasets/ToneStyle/TST2K) and [🤗 zrgong/PST50](https://huggingface.co/datasets/zrgong/PST50) evaluation datasets. See [`data/DOWNLOAD.md`](data/DOWNLOAD.md) for detailed instructions.

The evaluation metrics also require pretrained weights from [🤗 ToneStyle/MetricCkpt](https://huggingface.co/ToneStyle/MetricCkpt). See [`ckpt/DOWNLOAD.md`](ckpt/DOWNLOAD.md) for download instructions.

**Evaluation scripts**

`test_pst50.sh` and `test_tst2k.sh` run inference first and then compute four metrics corresponding to Section 4.1 of the paper:

- **CP (Content Preservation):** SSIM over LDC edge maps, measuring structural preservation.
- **ΔE (Color Difference):** CIEDE2000 perceptual color difference.
- **CD (Deep Color Difference):** CDFlow's learned color-difference metric, designed to align more closely with human preferences.
- **Aes (Aesthetic Quality):** aesthetic quality predicted by AesCLIP.

**Usage:** Edit the following five variables at the top of the evaluation script:

```bash
LORA_PATH=ckpt/ICTone-Fill-LoRA
FLUX_PATH="ckpt/FLUX-Fill"
GPUS="0,1,2,3"                 # Multiple GPUs, e.g. "0,1,2,3"; inference is sharded automatically
OUTPUT_DIR="eval/tst2k_test"
NUM=2000                       # Use 50 for PST50
```

Then run:

```bash
bash test_tst2k.sh             # Or: bash test_pst50.sh
```

**Outputs:**

- `${OUTPUT_DIR}/outputs/`: inference results (`NNNN.png`).
- `${OUTPUT_DIR}/run.shard*.log`: inference log for each GPU shard.
- `${OUTPUT_DIR}/eval_list.txt`: automatically generated four-column manifest (`content reference gt pred`).
- `${OUTPUT_DIR}/metrics/`:
  - `aes.txt`: AesCLIP aesthetic score.
  - `color_difference.txt`: CDFlow color difference.
  - `content_preserve.txt`: content preservation score.
  - `deltaE2000.txt`: ΔE2000 color difference.

## 🎪 Checklist

- ✅ Create the code repository and project page.
- ✅ Release the TST100K training dataset.
- ✅ Release the TST2K evaluation benchmark.
- ✅ Release the training, inference, and evaluation code.
- ✅ Release the ICTone LoRA weights.
- ✅ Launch the online Hugging Face demo.

## 📧 Contact and Citation

For questions or collaboration opportunities, please open an issue or contact:

- **Yuhai Deng:** `yhdeng.me@gmail.com`

If this project, dataset, or benchmark is useful to your research, please cite:

```bibtex
@misc{deng2026incontexttonestyletransfer,
  title         = {Towards In-Context Tone Style Transfer with A Large-Scale Triplet Dataset},
  author        = {Yuhai Deng and Huimin She and Wei Shen and Meng Li and Ruoxi Wu and Lunxi Yuan and Xiang Li},
  year          = {2026},
  eprint        = {2604.16114},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```

## 🙏 Acknowledgements

This codebase builds upon the following open-source projects:

- **Project implementation:** the in-context editing implementation from [ICEdit](https://github.com/River-Zhang/ICEdit).
- **Evaluation metrics:** [Neural Preset](https://github.com/ZHKKKe/NeuralPreset), [CDFlow](https://github.com/mergermarket/cdflow), and [AesCLIP](https://github.com/sxfly99/AesCLIP).

## 📜 License

- **Code:** non-commercial research use only. The code may be used, modified, and extended in academic research and other non-commercial settings. Commercial training, commercial fine-tuning, product integration, commercial system evaluation, resale, and commercial services are prohibited.
- **TST100K and TST2K datasets:** non-commercial academic and research use only. Users must also comply with the original licenses of all upstream datasets, including PPR10K, MIT-Adobe FiveK, Food-101, COCO, and Landscape HQ; the stricter terms take precedence. See [`data/TST100K/README.md`](data/TST100K/README.md) for details.
- **FLUX.1-Fill-dev base weights:** subject to the `flux-1-dev-non-commercial-license` published by Black Forest Labs.
