#!/usr/bin/env bash
# Shared inference + metrics driver for the test_*.sh wrappers.
#
# Positional args:
#   $1 dataset dir  (root of NNNN/{content,reference,gt}.png subdirs)
#                   -- ignored if TRIPLET_LIST env var is set.
#   $2 sample cap   (max subdirs / list rows to process)
#
# Required env:
#   LORA_PATH   path to the LoRA weights (dir with pytorch_lora_weights.safetensors)
#   FLUX_PATH   path to the FLUX-Fill checkpoint
#   GPUS        comma-separated CUDA_VISIBLE_DEVICES indices for inference
#   OUTPUT_DIR  where predictions + metrics are written
#
# Optional env:
#   TRIPLET_LIST  path to a 3-col TXT (content ref gt); when set, dataset dir is ignored.
#   METRIC_GPU (default 7), METRIC_LIST (auto-generated if missing),
#   METRIC_OUT_DIR ($OUTPUT_DIR/metrics), PY (python3.10),
#   SKIP_INFER, SKIP_METRICS.
set -euo pipefail

DATASET_DIR="$1"; NUM="$2"
: "${LORA_PATH:?LORA_PATH must be set}"
: "${FLUX_PATH:?FLUX_PATH must be set}"
: "${GPUS:?GPUS must be set (e.g. 4,5,6)}"
: "${OUTPUT_DIR:?OUTPUT_DIR must be set}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
IFS=',' read -r -a GPU_ARR <<< "$GPUS"
mkdir -p "${OUTPUT_DIR}"
if [ -n "${TRIPLET_LIST:-}" ]; then
    echo "[eval] triplet_list=${TRIPLET_LIST} num=${NUM} out=${OUTPUT_DIR} gpus=[${GPUS}]"
else
    echo "[eval] dataset=${DATASET_DIR} num=${NUM} out=${OUTPUT_DIR} gpus=[${GPUS}]"
fi
echo "[eval] lora=${LORA_PATH}  flux=${FLUX_PATH}"

# ---- Stage 1: inference (sharded over GPUS) ----
if [ -z "${SKIP_INFER:-}" ]; then
    if [ -n "${TRIPLET_LIST:-}" ]; then
        INPUT_ARGS=(--triplet-list "${TRIPLET_LIST}" --tst2k-num "${NUM}")
    else
        INPUT_ARGS=(--tst2k-dir "${DATASET_DIR}" --tst2k-num "${NUM}")
    fi
    pids=()
    for i in "${!GPU_ARR[@]}"; do
        gpu="${GPU_ARR[$i]}"
        CUDA_VISIBLE_DEVICES="${gpu}" python inference.py \
            "${INPUT_ARGS[@]}" \
            --flux-path "${FLUX_PATH}" --lora-path "${LORA_PATH}" \
            --output-dir "${OUTPUT_DIR}" \
            --image-size 512 --lut-size 33 \
            --num-inference-steps 4 --guidance-scale 50.0 \
            --seed 666 --dtype bfloat16 --generator-device cuda \
            --num-shards "${#GPU_ARR[@]}" --shard-index "${i}" \
            > "${OUTPUT_DIR}/run.shard${i}_gpu${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    fail=0; for pid in "${pids[@]}"; do wait "${pid}" || fail=1; done
    [ "${fail}" -ne 0 ] && { echo "[eval] shard failed — see ${OUTPUT_DIR}/run.shard*.log"; exit 1; }
fi

[ -n "${SKIP_METRICS:-}" ] && exit 0

# ---- Stage 2: metrics (auto-generate eval list, then run 4 metrics) ----
METRICS_DIR="src/metrics"
METRIC_LIST="${METRIC_LIST:-${OUTPUT_DIR}/eval_list.txt}"
METRIC_OUT_DIR="${METRIC_OUT_DIR:-${OUTPUT_DIR}/metrics}"
METRIC_GPU="${METRIC_GPU:-7}"
PY="${PY:-python3.10}"
mkdir -p "${METRIC_OUT_DIR}"

# Auto-generate 4-column list [content ref gt pred] mirroring inference.py
# ordering. Rows without gt are dropped. Source is TRIPLET_LIST when set, else
# the NNNN/{content,reference,gt}.png subdirs under DATASET_DIR.
if [ ! -f "${METRIC_LIST}" ]; then
    DATASET_DIR="${DATASET_DIR}" OUTPUT_DIR="${OUTPUT_DIR}" \
    NUM="${NUM}" METRIC_LIST="${METRIC_LIST}" \
    TRIPLET_LIST="${TRIPLET_LIST:-}" ${PY} - <<'PY'
import os
from pathlib import Path
pred_dir = Path(os.environ["OUTPUT_DIR"]) / "outputs"
cap = int(os.environ["NUM"])
out = Path(os.environ["METRIC_LIST"])
out_base = out.resolve().parent
triplet_list = os.environ.get("TRIPLET_LIST") or ""
rows, kept = [], 0

def fmt(path):
    return Path(os.path.relpath(path.resolve(), start=out_base)).as_posix()

if triplet_list:
    list_path = Path(triplet_list)
    list_base = list_path.resolve().parent

    def resolve_list_path(value):
        path = Path(value)
        return path if path.is_absolute() else list_base / path

    with open(list_path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            c, r = resolve_list_path(parts[0]), resolve_list_path(parts[1])
            g = resolve_list_path(parts[2]) if len(parts) >= 3 else None
            if not (c.exists() and r.exists()):
                continue
            gi, kept = kept, kept + 1
            if g is not None and g.exists():
                rows.append(
                    f"{fmt(c)} {fmt(r)} {fmt(g)} "
                    f"{fmt(pred_dir / f'{gi:04d}.png')}"
                )
            if kept >= cap:
                break
else:
    root = Path(os.environ["DATASET_DIR"])
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        c, r = sub / "content.png", sub / "reference.png"
        if not (c.exists() and r.exists()):
            continue
        gi, kept = kept, kept + 1
        g = sub / "gt.png"
        if g.exists():
            rows.append(
                f"{fmt(c)} {fmt(r)} {fmt(g)} "
                f"{fmt(pred_dir / f'{gi:04d}.png')}"
            )
        if kept >= cap:
            break
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(rows) + ("\n" if rows else ""))
print(f"[metrics] wrote {len(rows)} rows (kept {kept}, cap={cap})")
PY
fi

launch() {  # $1=cuda $2=script.py $3=basename (log/txt stem) $4..=extra args
    local cuda="$1" script="$2" base="$3"; shift 3
    CUDA_VISIBLE_DEVICES="${cuda}" ${PY} "${METRICS_DIR}/${script}" \
        --list "${METRIC_LIST}" --out "${METRIC_OUT_DIR}/${base}.txt" "$@" \
        > "${METRIC_OUT_DIR}/${base}.log" 2>&1 &
    pids+=("$!")
}

pids=()
launch "${METRIC_GPU}" aes.py              aes              --mode zs
launch "${METRIC_GPU}" color_difference.py color_difference --batch_size 8 --num_workers 4
launch "${METRIC_GPU}" content_preserve.py content_preserve
launch ""              DeltaE2000.py       deltaE2000

fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "${fail}" -ne 0 ] && { echo "[metrics] failed — see ${METRIC_OUT_DIR}/*.log"; exit 1; }
echo "[eval] done — results in ${METRIC_OUT_DIR}"
