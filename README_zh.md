[English](README.md) | **简体中文**

# [ECCV' 2026] Towards In-Context Tone Style Transfer with a Large-Scale Triplet Dataset

<!-- **Towards In-Context Tone Style Transfer with a Large-Scale Triplet Dataset** -->
<div align="center">
  <a href="https://arxiv.org/pdf/2604.16114"><img src="https://img.shields.io/badge/arXiv-2604.16114-b31b1b.svg" alt="Paper"></a>
  <a href="https://dengyuhai.github.io/ICTone_Project/"><img src="https://img.shields.io/badge/Project%20Page-Visit-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/spaces/ToneStyle/ICTone-Fill"><img src="https://img.shields.io/badge/%F0%9F%A4%97-HF%20Demo-yellow.svg" alt="Hugging Face Demo"></a>
  <br>
  <a href="https://huggingface.co/ToneStyle/ICTone-Fill-LoRA"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Model%20Weights-green.svg" alt="Model Weights"></a>
  <a href="https://huggingface.co/datasets/ToneStyle/TST100K"><img src="https://img.shields.io/badge/%F0%9F%A4%97-TST100K%20Dataset-blue.svg" alt="TST100K Dataset"></a>
  <a href="https://huggingface.co/datasets/ToneStyle/TST2K"><img src="https://img.shields.io/badge/%F0%9F%A4%97-TST2K%20Benchmark-blueviolet.svg" alt="TST2K Benchmark"></a>
</div>

Yuhai Deng<sup>&#42;</sup>, Huimin She<sup>&#42;</sup>, Wei Shen<sup>†</sup>, Meng Li, Ruoxi Wu, Lunxi Yuan, Xiang Li<sup>✉</sup>

<sup>&#42;</sup> 共同第一作者 · <sup>†</sup> 项目Leader · <sup>✉</sup> 通讯作者

Nankai University · OPPO AI Center


## 📝 概述

![ICTone teaser](assets/teaser.jpg)

**ICTone** 将 **参考式色调风格迁移 (Reference-based Tone Style Transfer)** 建模为一个 **in-context generation** 任务：与传统方法用两条独立编码器分别抽取 content / reference 特征、再在 decoder 里融合不同，ICTone 直接把 content 和 reference 作为 **联合上下文** 送入 diffusion transformer，让模型利用生成大模型的语义先验去做语义感知的色调迁移，从根本上缓解了先前方法常见的「不当颜色迁移 / 语义错位」问题（例如把栏杆的暖色错误迁移到人脸上）。

## 📮 最新动态

- **[2026.08.15]** ICTone Hugging Face 在线 Demo 已上线：[🤗 ICTone-Fill Demo](https://huggingface.co/spaces/ToneStyle/ICTone-Fill)。
- **[2026.08.14]** LoRA权重已发布， 可直接下载推理[🤗 ToneStyle/ICTone-Fill-LoRA](https://huggingface.co/ToneStyle/ICTone-Fill-LoRA)。
- **[2026.08.13]** 大规模三元组数据集（100k+ content / reference / gt） **TST100K** 现已开源 [🤗 ToneStyle/TST100K](https://huggingface.co/datasets/ToneStyle/TST100K)。
- **[2026-07.15]** **TST2K** Benchmark 现已开源 [🤗 ToneStyle/TST2K](https://huggingface.co/datasets/ToneStyle/TST2K)，覆盖 portrait / food / landscape / night / lifestyle。
- **[2026.06.18]** ICTone 被 ECCV 2026 接收。

## 💻 环境配置

推荐使用 conda 创建独立环境。实测环境为 **Python 3.10 + CUDA GPU (支持 BF16)**。

```bash
# 1. 创建并激活环境
conda create -n ictone python=3.10 -y
conda activate ictone

# 2. 安装依赖
cd ICTone
pip install -r requirements.txt
```

## 🚀 快速推理

### 🖼️ 单对图片推理

```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --content   assets/example_content.png \
    --reference assets/example_reference.png \
    --flux-path black-forest-labs/FLUX.1-Fill-dev \
    --lora-path ToneStyle/ICTone-Fill-LoRA \
    --output-file ./output/example_output.png \
    --num-inference-steps 4 \
    --guidance-scale 50.0 \
    --seed 666
```


## 🛠️ 训练与评测

### 🔧 训练

**数据**：`data/TST100K/`（100k+ 三元组，`content_images/` + `style_images/` + `triplet.json`），从 [🤗 ToneStyle/TST100K](https://huggingface.co/datasets/ToneStyle/TST100K) 下载。详细下载命令见 [`data/DOWNLOAD.md`](data/DOWNLOAD.md)。

**配置**：`configs/ictone_pair_lora.yaml`

- `flux_path`：FLUX-Fill 目录。默认相对路径为 `ckpt/FLUX-Fill`，需在 `ICTone/` 目录下启动训练/推理；也可以填写 Hugging Face 模型仓库 ID。
- `train.dataset.triplet_json` / `data_root` ：三元组清单、根目录。
- `train.max_steps / save_interval / sample_interval / keep_last_k`：训练最大步数、权重保存间隔、训练可视化间隔。使用 4×A100 训练 50k 步，`lr=1e-4`，`weight_decay=1e-3`。
- `train.ictone_val_*`：训练中定期在 `data/TST2K` 上的可视化验证配置。

**启动**：

```bash
# 单卡 smoke test
GPUS=0 bash train_ictone.sh

# 默认4卡
bash train_ictone.sh

```

支持的环境变量：`GPUS / NPROC / CONFIG / PORT / RUN_TAG / LORA_PATH / MAX_STEPS / DEBUG`。日志与产物写入 `runs/<STAMP>_<RUN_TAG>/`（含 `train.log`、`ckpt/`、`validate/`、`config.yaml`）。WanDB 默认禁用。

### 📊 评测
**数据准备**

需要先下载测试数据 [🤗 ToneStyle/TST2K](https://huggingface.co/datasets/ToneStyle/TST2K) 和 [🤗 zrgong/PST50](https://huggingface.co/datasets/zrgong/PST50)。详细下载命令见 [`data/DOWNLOAD.md`](data/DOWNLOAD.md)。

还需要下载指标所需的模型权重[🤗 ToneStyle/MetricCkpt](https://huggingface.co/ToneStyle/MetricCkpt)。详细下载命令见 [`ckpt/DOWNLOAD.md`](ckpt/DOWNLOAD.md)

**测试启动脚本**

`test_pst50.sh` 和 `test_tst2k.sh` 是一键脚本：先做推理，再算 4 个指标。**评测指标**对应论文 §4.1：

- **CP (Content Preservation)** — 用 LDC 边缘图上的 SSIM，衡量结构保持度。
- **ΔE (Color Difference)** — CIEDE2000，衡量感知色差。
- **CD (Deep Color Difference)** — CDFlow 学习式色差，与人类偏好更一致。
- **Aes (Aesthetic Quality)** — AesCLIP 输出的美学分。

**用法**：编辑脚本顶部 5 个变量

```bash
LORA_PATH=ckpt/ICTone-Fill-LoRA
FLUX_PATH="ckpt/FLUX-Fill"
GPUS="0,1,2,3"                    # 多卡如 "0,1,2,3"，推理自动分片
OUTPUT_DIR="eval/tst2k_test"
NUM=2000                      # PST50 用 50
```

然后：

```bash
bash test_tst2k.sh          # 或 bash test_pst50.sh
```

**产物**：

- `${OUTPUT_DIR}/outputs/`：推理结果（`NNNN.png`）。
- `${OUTPUT_DIR}/run.shard*.log`：每张卡的推理日志。
- `${OUTPUT_DIR}/eval_list.txt`：自动生成的 `[content ref gt pred]` 4 列清单。
- `${OUTPUT_DIR}/metrics/`：
  - `aes.txt` — 美学分（AesCLIP）
  - `color_difference.txt` — CDFlow 色差
  - `content_preserve.txt` — 内容保持度
  - `deltaE2000.txt` — ΔE2000


## 🎪 项目清单

- ✅ 创建代码仓库与项目主页
- ✅ 开源 TST100K 训练数据集
- ✅ 开源 TST2K 评测基准
- ✅ 开源训练、推理、评测代码
- ✅ 开源 ICTone LoRA 权重
- ✅ 上线 Hugging Face 在线 Demo

## 📧 联系方式与引用

如有问题或合作意向，欢迎通过 Issue 联系，或直接联系：

- **Yuhai Deng**：`yhdeng.me@gmail.com`

如果本项目 / 数据集 / 评测基准对你有帮助，请引用：

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

## 🙏 致谢

本代码建立在以下开源工作之上，特此致谢：

- **项目代码**：参考了 [ICEdit](https://github.com/River-Zhang/ICEdit) 的 in-context editing 实现。
- **指标评测**：感谢 [Neural Preset](https://github.com/ZHKKKe/NeuralPreset)、[CDFlow](https://github.com/mergermarket/cdflow)、[AesCLIP](https://github.com/sxfly99/AesCLIP) 提供的开源实现与权重。

## 📜 许可证

- 代码：**Non-commercial research use only**。允许在学术研究和非商业环境下使用、修改和二次开发；商业训练 / 商业微调 / 产品化 / 商业系统评测 / 转售 / 商业服务均不允许。
- 数据集 **TST100K** / **TST2K**：同样为 **non-commercial academic and research use only**。使用者需自行遵守 PPR10K、MIT-Adobe FiveK、Food-101、COCO、Landscape HQ 等上游数据集的原始 License（以更严格者为准），详见 [`data/TST100K/README.md`](data/TST100K/README.md)。
- 基座权重 **FLUX.1-Fill-dev**：遵循 Black Forest Labs 发布的 `flux-1-dev-non-commercial-license`。
