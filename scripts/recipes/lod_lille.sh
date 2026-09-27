#!/usr/bin/env bash
# lod-lille-0.6B: Qwen3-0.6B-Base + LoRA r32 (alpha 64), independent options and a
# comparison layer, 32k budget, trained on lod-corpus-distilled -- the corpus with
# lod-stor-4B's distribution on every train question (`build_corpus.py distill`) -- at
# distillation alpha 0.5; lr 2e-4, 8,000 steps, 98,304-token batches.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKBONE=Qwen/Qwen3-0.6B-Base LORA_R=32 LORA_ALPHA=64 MAX_BATCH_TOKENS=98304 \
  EXTRA="--distill-alpha 0.5" \
  bash "$HERE/release.sh" lod-corpus-distilled lod-lille 2e-4 8000
