# Datasets

ICTone ships with three datasets. Download the ones you need — commands below are meant to be **copy-pasted individually**, not run in one shot.

> All commands should be executed **from the `ICTone/` directory**. Optional: `export HF_HUB_ENABLE_HF_TRANSFER=1` to speed up large-file downloads.

## Overview

| Name | Repo | Used for | Access |
|---|---|---|---|
| `TST100K` | [ToneStyle/TST100K](https://huggingface.co/datasets/ToneStyle/TST100K) | Training triplets (100k+, large) | Public |
| `TST2K` | [ToneStyle/TST2K](https://huggingface.co/datasets/ToneStyle/TST2K) | Evaluation benchmark (2k) | Public |
| `PST50` | [zrgong/PST50](https://huggingface.co/datasets/zrgong/PST50) | Evaluation set (50) | Public |

---

## 1) TST2K (evaluation benchmark, small)

```bash
huggingface-cli download --repo-type dataset ToneStyle/TST2K \
    --local-dir data/TST2K \
    --local-dir-use-symlinks False
```

## 2) PST50 (evaluation set, small)

```bash
huggingface-cli download --repo-type dataset zrgong/PST50 \
    --local-dir data/PST50 \
    --local-dir-use-symlinks False
```

Raw PST50 uses PPR10K-style naming (`content_log/in{i}.png`, `paired_style/tar{i}.png`, `paired_gt/gt{i}.png`) instead of the `NNNN/{content,reference,gt}.png` layout used by TST2K. Rather than reshuffling files, generate a 3-column triplet list (`content ref gt`) that `inference_tone.py --triplet-list` and `_run_eval.sh` can read directly:

```bash
# from ICTone/
python data/make_pst50_list.py                          # default: content_log as source
# or use content_709 instead:
# python data/make_pst50_list.py --content-src content_709
```

This writes `data/PST50.txt` (50 rows). `test_pst50.sh` already points at this file via the `TRIPLET_LIST` variable.

## 3) TST100K (training data, large — only needed for training)

```bash
huggingface-cli download --repo-type dataset ToneStyle/TST100K \
    --local-dir data/TST100K \
    --local-dir-use-symlinks False
```
