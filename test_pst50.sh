#!/usr/bin/env bash
# PST50 end-to-end evaluation. Edit the assignments below as needed.
# See _run_eval.sh for optional env vars (METRIC_GPU, SKIP_INFER, ...).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LORA_PATH="ckpt/ICTone-Fill-LoRA"
FLUX_PATH="ckpt/FLUX-Fill"
GPUS="0"
OUTPUT_DIR="eval/pst50"
NUM=50

# Use a 3-col triplet list (content ref gt) instead of an NNNN/ dir tree.
# Generate it once with: python data/make_pst50_list.py
TRIPLET_LIST="data/PST50.txt"

LORA_PATH="${LORA_PATH}" FLUX_PATH="${FLUX_PATH}" GPUS="${GPUS}" \
OUTPUT_DIR="${OUTPUT_DIR}" TRIPLET_LIST="${TRIPLET_LIST}" \
    exec ./_run_eval.sh "" "${NUM}"
