#!/usr/bin/env bash
# TST2K end-to-end evaluation. Edit the four assignments below as needed.
# See _run_eval.sh for optional env vars (METRIC_GPU, SKIP_INFER, ...).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

LORA_PATH="ckpt/ICTone-Fill-LoRA"
FLUX_PATH="ckpt/FLUX-Fill"
GPUS="0,1,2,3"
OUTPUT_DIR="eval/tst2k"
NUM=2000

LORA_PATH="${LORA_PATH}" FLUX_PATH="${FLUX_PATH}" GPUS="${GPUS}" OUTPUT_DIR="${OUTPUT_DIR}" \
    exec ./_run_eval.sh data/TST2K "${NUM}"

