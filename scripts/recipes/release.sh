#!/usr/bin/env bash
# The release recipe for one model: train, attach the confidence head, write the report,
# the gates and the model card. The released models ship at temperature 1 (no fitted
# temperature), so nothing here fits one; devood is scored, never fitted.
#
# Usage: scripts/recipes/release.sh <corpus> <run> <lr> <steps>
#   <corpus>  a corpus under $LOD_ROOT/data/ (train, val, devood, devood_sel, devreal,
#             testreal .jsonl and meta.json; built by scripts/build_corpus.py)
#   <run>     writes $LOD_ROOT/runs/<run> (training) and $LOD_ROOT/runs/<run>-release
#             (the released checkpoint, report/ and MODEL_CARD.md)
# Knobs (environment): BACKBONE LORA_R LORA_ALPHA MAX_STATE MAX_TOTAL MAX_BATCH_TOKENS
#   SEED EXTRA (extra train.py flags). Every step is skipped when its output exists, so a
#   relaunch after preemption resumes; training itself resumes from its last step-*.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${LOD_ROOT:-.}"
CORPUS="${1:?usage: release.sh <corpus> <run> <lr> <steps>}"
NAME="${2:?usage: release.sh <corpus> <run> <lr> <steps>}"
LR="${3:?usage: release.sh <corpus> <run> <lr> <steps>}"
STEPS="${4:?usage: release.sh <corpus> <run> <lr> <steps>}"
DATA="$ROOT/data/$CORPUS"; RUN="$ROOT/runs/$NAME"; REL="$RUN-release"; OUT="$REL/report"
BACKBONE="${BACKBONE:-Qwen/Qwen3-0.6B-Base}"
MS="${MAX_STATE:-30000}"; MT="${MAX_TOTAL:-32768}"; MBT="${MAX_BATCH_TOKENS:-98304}"
EXTRA="${EXTRA:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
py() { local script="$1"; shift; uv run --project "$REPO" python "$REPO/scripts/$script" "$@"; }

for f in train.jsonl val.jsonl devood.jsonl devood_sel.jsonl devreal.jsonl testreal.jsonl; do
  [ -f "$DATA/$f" ] || { echo "release: no $DATA/$f" >&2; exit 2; }
done

# ---- 1. train: independent options + comparison layer, 32k budget, best/ on devood_sel
if [ ! -f "$RUN/last/lod.json" ]; then
  # shellcheck disable=SC2086
  py train.py \
    --backbone "$BACKBONE" --lora-r "${LORA_R:-32}" --lora-alpha "${LORA_ALPHA:-64}" \
    --weights-dtype bf16 --amp bf16 --option-mode independent \
    --mixer-dim 256 --mixer-heads 4 --mixer-topk 64 --stage1-weight 0.25 --heads 1 \
    --batch-size 32 --bucket 32 --max-batch-tokens "$MBT" \
    --max-state-tokens "$MS" --max-tokens "$MT" --attn flex --grad-ckpt --compile \
    --epochs 2 --max-steps "$STEPS" --lr "$LR" --lr-head 1e-3 --warmup 300 \
    --eval-every 1000 --save-every 1000 --keep-last 2 --patience 0 --no-progress \
    --seed "${SEED:-0}" --criteria-augment --num-workers 16 \
    --data "$DATA" --devood "$DATA/devood_sel.jsonl" --resume auto --out "$RUN" $EXTRA \
    || { echo "train.py failed; stopping" >&2; exit 1; }
fi

# ---- 2. the released checkpoint: best/ at T = 1, plus the confidence head trained on
#         exactly that distribution (the scorer is frozen; the head reads its output)
python3 - "$RUN/best/lod.json" <<'PY' || { echo "release: best/ does not carry T = 1" >&2; exit 1; }
import json, sys
cfg = json.load(open(sys.argv[1]))
sys.exit(0 if cfg["temperature"] == 1.0 and cfg.get("temperature_logn", 0.0) == 0.0 else 1)
PY
[ -f "$REL/lod.json" ] || py train_confidence.py --ckpt "$RUN/best" --data "$DATA" \
  --criteria-augment --epochs 8 --lr 1e-3 --batch-size 32 --num-workers 12 \
  --max-batch-tokens "$MBT" --ood "$DATA/devood.jsonl" \
  --features log_n,max_p,entropy_ratio,margin,logit_gap --out "$REL" \
  > "$RUN.confidence.txt" 2>&1 || { echo "train_confidence.py failed" >&2; exit 1; }
echo "######## released checkpoint: $REL"

# ---- 3. report, read on exactly what ships
mkdir -p "$OUT"
EV="--attn flex --max-batch-tokens $MBT --no-progress"
[ -s "$OUT/independence.txt" ] || py check_independence.py --ckpt "$REL" --device cpu \
  > "$OUT/independence.txt" 2>&1
for sp in devood devreal testreal; do
  extra=""; [ "$sp" = testreal ] && extra="--oracle-temperature"
  # shellcheck disable=SC2086
  [ -s "$OUT/eval-$sp.jsonl" ] || py evaluate.py --ckpt "$REL" --data "$DATA/$sp.jsonl" \
    $EV $extra --out "$OUT/eval-$sp.jsonl" > "$OUT/eval-$sp.txt" 2>&1
  grep -E "^MICRO" "$OUT/eval-$sp.txt" | tail -1 | sed "s/^/######## $sp /"
done
py report.py ci "$OUT/eval-testreal.jsonl" --json "$OUT/per_source.json" \
  > "$OUT/per_source.md" 2>&1 || true
[ -s "$OUT/probes.json" ] || py gates.py probes --ckpt "$REL" --json "$OUT/probes.json" \
  > "$OUT/probes.txt" 2>&1
# real vs code-labelled by task-name prefix, as the released gates were read; add
# --corpus "$DATA" to split by the corpus meta.json instead
py gates.py summary --ckpt "$REL" --report "$OUT" > "$OUT/gates.json" \
  || echo "######## blocking gates failed (see $OUT/gates.json)"

KNOWLEDGE="$ROOT/data/probes/knowledge.jsonl"
if [ -f "$KNOWLEDGE" ] && [ ! -s "$OUT/knowledge.jsonl" ]; then
  # shellcheck disable=SC2086
  py evaluate.py --ckpt "$REL" --data "$KNOWLEDGE" $EV --out "$OUT/knowledge.jsonl" \
    > "$OUT/knowledge.txt" 2>&1 || true
fi
if [ -f "$DATA/depth_check.jsonl" ] && [ ! -s "$OUT/depth_check.jsonl" ]; then
  # shellcheck disable=SC2086
  py evaluate.py --ckpt "$REL" --data "$DATA/depth_check.jsonl" $EV \
    --out "$OUT/depth_check.jsonl" > "$OUT/depth_check.txt" 2>&1 || true
  py report.py depth "$OUT/depth_check.jsonl" > "$OUT/depth_report.txt" 2>&1 || true
fi
[ -s "$OUT/regression.json" ] || py regression_suite.py --ckpt "$REL" --corpus "$DATA" \
  --json "$OUT/regression.json" > "$OUT/regression.txt" 2>&1 || true
py report.py by-domain "$OUT/eval-testreal.jsonl" --corpus "$DATA" \
  --json "$OUT/by_domain.json" > "$OUT/by_domain.txt" 2>&1 || true
py model_card.py --ckpt "$REL" --report "$OUT" --run "$RUN" --corpus "$DATA" \
  --name "$NAME" --out "$REL/MODEL_CARD.md" || true
echo "######## release done: $REL"
