#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)"

# ── Config ────────────────────────────────────────────────────────────────────
PRETRAIN_CFG="vljepa/configs/pretrain_vg.yaml"
PRETRAIN_OUT="outputs/pretrain_vg"
MANIFEST="data/vsibench_manifest.jsonl"
VSI_DIR="/home/edisk/Dataset/VSI-Bench"
RESULTS="outputs/pretrain_vg/vsibench_results.jsonl"
BATCH_SIZE=32
NUM_WORKERS=8

# ── Stage 1: Pretrain ─────────────────────────────────────────────────────────
echo "==> [1/3] Pretraining..."
python scripts/train_pretrain.py --config "$PRETRAIN_CFG"

# ── Find latest checkpoint ────────────────────────────────────────────────────
CKPT=$(ls -t "$PRETRAIN_OUT"/step_*.pt 2>/dev/null | head -1)
if [[ -z "$CKPT" ]]; then
    echo "ERROR: no checkpoint found in $PRETRAIN_OUT" >&2
    exit 1
fi
echo "==> Using checkpoint: $CKPT"

# ── Stage 2: Build eval manifest (skip if exists) ─────────────────────────────
echo "==> [2/3] Building VSI-Bench manifest..."
if [[ ! -f "$MANIFEST" ]]; then
    python scripts/make_vsibench_manifest.py \
        --vsi-dir "$VSI_DIR" \
        --out "$MANIFEST" \
        --mode mc
else
    echo "  manifest exists, skipping."
fi

# ── Stage 3: Inference + Accuracy ─────────────────────────────────────────────
echo "==> [3/3] Running inference..."
python scripts/run_inference.py \
    --config vljepa/configs/inference.yaml \
    --checkpoint "$CKPT" \
    --manifest "$MANIFEST" \
    --mode discriminative_vqa \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    > "$RESULTS"

echo "==> Computing accuracy..."
python - <<'EOF'
import json, sys
from pathlib import Path
from collections import defaultdict

manifest = [json.loads(l) for l in open("data/vsibench_manifest.jsonl")]
results  = [json.loads(l) for l in open("outputs/pretrain_vg/vsibench_results.jsonl")]

by_type = defaultdict(lambda: [0, 0])  # correct, total
for gt_row, pred_row in zip(manifest, results):
    qt = gt_row["question_type"]
    correct = int(gt_row["target"] == pred_row["prediction"])
    by_type[qt][0] += correct
    by_type[qt][1] += 1

total_c = sum(v[0] for v in by_type.values())
total_t = sum(v[1] for v in by_type.values())
print(f"\nOverall accuracy: {total_c}/{total_t} = {total_c/total_t*100:.1f}%\n")
print(f"{'Question Type':<35} {'Acc':>6}  (n)")
print("-" * 50)
for qt, (c, t) in sorted(by_type.items()):
    print(f"{qt:<35} {c/t*100:>5.1f}%  ({t})")
EOF

echo "==> Done. Results saved to $RESULTS"
