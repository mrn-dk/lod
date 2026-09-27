#!/usr/bin/env bash
# lod-stor-4B: Qwen3-4B-Base + LoRA r64 (alpha 128), independent options and a comparison
# layer, 32k budget, trained on lod-corpus; lr 1e-4, 8,000 steps, 49,152-token batches
# (the cap that fits a 96 GB card at this size).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKBONE=Qwen/Qwen3-4B-Base LORA_R=64 LORA_ALPHA=128 MAX_BATCH_TOKENS=49152 \
  bash "$HERE/release.sh" lod-corpus lod-stor 1e-4 8000
