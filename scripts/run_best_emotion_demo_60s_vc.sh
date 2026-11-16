#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERR] line=$LINENO cmd=$BASH_COMMAND exit=$?"; exit 1' ERR

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

# ✅ 你的中文 5s 音色参考
REF_WAV="${REF_WAV:-/data/lizhen/EmoVoice/data/speakers/lizhen_ch_neutral.wav}"

COSYVOICE_REPO="${COSYVOICE_REPO:-$ROOT/third_party/CosyVoice}"
COSYVOICE3_MODEL="${COSYVOICE3_MODEL:-$ROOT/ckpts/CosyVoice/Fun-CosyVoice3.0}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT/outputs/demos/${TS}_best_demo"
mkdir -p "$RUN_ROOT/inputs"

JSONL="$RUN_ROOT/inputs/story60.jsonl"
python tools/build_story60_best_jsonl.py --ref_wav "$REF_WAV" --out_jsonl "$JSONL" --key_prefix "demo"

echo "[INFO] ===== Stage1: CosyVoice3 instruct2 (emotion zh) ====="
STAGE1="$RUN_ROOT/stage1_instruct"
mkdir -p "$STAGE1"
python -u tools/cosyvoice3_batch.py \
  --cosyvoice_repo "$COSYVOICE_REPO" \
  --model_dir "$COSYVOICE3_MODEL" \
  instruct2 \
  --jsonl "$JSONL" \
  --prompt_wav "$REF_WAV" \
  --out_dir "$STAGE1" \
  --concat --no_segments 2>&1 | tee "$RUN_ROOT/stage1.log"

echo "[INFO] ===== Stage2: CosyVoice3 VC (timbre transfer) ====="
STAGE2="$RUN_ROOT/stage2_vc"
mkdir -p "$STAGE2"
python -u tools/cosyvoice3_batch.py \
  --cosyvoice_repo "$COSYVOICE_REPO" \
  --model_dir "$COSYVOICE3_MODEL" \
  vc \
  --src_dir "$STAGE1" \
  --prompt_wav "$REF_WAV" \
  --out_dir "$STAGE2" \
  --pattern "demo_*.wav" \
  --concat --no_segments 2>&1 | tee "$RUN_ROOT/stage2.log"

echo "[INFO] ===== Stage3: concat + trim + normalize + crossfade ====="
FINAL="$RUN_ROOT/final_demo_60s.wav"
python tools/concat_story_best.py \
  --in_dir "$STAGE2" \
  --out_wav "$FINAL" \
  --pattern "demo_*.wav" \
  --sr 24000 \
  --fade_ms 25 \
  --gap_ms 30

echo "[DONE] RUN_ROOT=$RUN_ROOT"
echo "[DONE] FINAL=$FINAL"