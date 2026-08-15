#!/bin/bash
# Local launch script for pair-driven ICTone LoRA training.
#
# Defaults to a multi-GPU run using the merged 100k pair config
# (unify_sampled_pairs_A -> ictone_pair_lora.yaml). Meant for local
# dev / smoke tests / short debug runs. For a full 7-GPU run just override
# GPUS.
#
# Usage
# -----
#   # Default GPUs (see GPUS below):
#   bash ICTone/train_ictone.sh
#
#   # Single GPU (debug / smoke):
#   GPUS=0 bash ICTone/train_ictone.sh
#
#   # Pick a subset:
#   GPUS=0,1,2,3 bash ICTone/train_ictone.sh
#
#   # Override config or port:
#   CONFIG=ictone_pair_lora.yaml PORT=41354 \
#       bash ICTone/train_ictone.sh
#
# Env vars
# --------
#   GPUS      comma-separated CUDA device list (default: 0,1,2,3)
#   NPROC     number of processes (default: inferred from GPUS)
#   CONFIG    config file under ICTone/configs (default: ictone_pair_lora.yaml)
#   PORT      accelerate main-process port (default: 41353)
#   RUN_TAG   suffix appended to WANDB run name / save dir (default: local)
#   LORA_PATH LoRA warm-start ckpt (dir or .safetensors); overrides yaml.
#             Set to empty to skip warm-start and fall back to the yaml value
#             (which defaults to null = from-scratch).
#   MAX_STEPS override train.max_steps for a quick run (default: unset)
#   DEBUG     if set, adds -X faulthandler and CUDA_LAUNCH_BLOCKING=1

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

CONFIG="${CONFIG:-ictone_pair_lora.yaml}"
PORT="${PORT:-41353}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-$(echo "$GPUS" | tr ',' '\n' | grep -c .)}"
# RUN_TAG becomes the suffix of the log/save dir:
#   runs/<STAMP>_<RUN_TAG>/train.log
# Bump this whenever the dataset changes so old runs stay clearly separated.
RUN_TAG="${RUN_TAG:-v2cct}"

# LoRA warm-start path. Overrides ``train.lora_path`` in the yaml via
# ``XFL_LORA_PATH`` (see get_config in src/train/train.py). Use an empty
# string (or the literals ``None`` / ``null``) to skip warm-start and fall
# back to the yaml value (which defaults to null = from-scratch).
LORA_PATH="${LORA_PATH-}"


CONFIG_PATH="configs/${CONFIG}"
if [ ! -f "$CONFIG_PATH" ]; then
    echo "[fatal] config not found: $CONFIG_PATH" >&2
    exit 1
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export XFL_CONFIG="$CONFIG_PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="."
export WANDB_DISABLED=true
export WANDB_MODE=disabled
export NCCL_ASYNC_ERROR_HANDLING=1

# Treat "", "None", "null" (case-insensitive) all as "no warm-start".
_lora_lower="$(printf '%s' "$LORA_PATH" | tr '[:upper:]' '[:lower:]')"
if [ -n "$LORA_PATH" ] && [ "$_lora_lower" != "none" ] && [ "$_lora_lower" != "null" ]; then
    export XFL_LORA_PATH="$LORA_PATH"
    echo "[launch] LoRA warm-start: $LORA_PATH"
else
    echo "[launch] LoRA warm-start: disabled (yaml default = null = from-scratch)"
fi
unset _lora_lower

if [ -n "${MAX_STEPS:-}" ]; then
    export XFL_MAX_STEPS="$MAX_STEPS"   # picked up only if train.py wires it
    echo "[launch] MAX_STEPS override requested: $MAX_STEPS (see train.py to honor XFL_MAX_STEPS)"
fi

DEBUG_ARGS=""
if [ -n "${DEBUG:-}" ]; then
    export CUDA_LAUNCH_BLOCKING=1
    DEBUG_ARGS="-X faulthandler"
    echo "[launch] DEBUG mode: CUDA_LAUNCH_BLOCKING=1, faulthandler on"
fi

MULTI_GPU_FLAG=""
if [ "$NPROC" -gt 1 ]; then
    MULTI_GPU_FLAG="--multi_gpu"
fi

echo "[launch] cwd=$(pwd)"
echo "[launch] CONFIG=$CONFIG_PATH"
echo "[launch] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  num_processes=$NPROC"
echo "[launch] port=$PORT  run_tag=$RUN_TAG"

# Log to runs/<date>_<tag>/train.log while tee'ing to stdout.
# Export the same name as ``XFL_RUN_NAME`` so train.py writes its
# checkpoints / samples / config.yaml under the identical directory
# (otherwise train.py generates its own timestamp and outputs get split).
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="runs/${STAMP}_${RUN_TAG}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train.log"
export XFL_RUN_NAME="${STAMP}_${RUN_TAG}"
echo "[launch] log=$LOG_FILE"
echo "[launch] run_name=$XFL_RUN_NAME"

set -o pipefail
accelerate launch \
    $MULTI_GPU_FLAG \
    --num_processes "$NPROC" \
    --num_machines 1 \
    --main_process_port "$PORT" \
    --mixed_precision bf16 \
    $DEBUG_ARGS \
    -m src.train.train 2>&1 | tee "$LOG_FILE"
