#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERR] line=$LINENO cmd=$BASH_COMMAND exit=$?"; exit 1' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ====== 你提供的中文参考音色（5s）=====
REF_WAV="${REF_WAV:-/data/lizhen/EmoVoice/data/speakers/lizhen_ch_neutral.wav}"

# ====== CosyVoice3 路径 ======
COSYVOICE_REPO="${COSYVOICE_REPO:-$ROOT_DIR/third_party/CosyVoice}"
COSYVOICE3_MODEL="${COSYVOICE3_MODEL:-$ROOT_DIR/ckpts/CosyVoice/Fun-CosyVoice3.0}"

# ====== 拼接参数（减少卡顿）=====
GAP_MS="${GAP_MS:-80}"
FADE_MS="${FADE_MS:-12}"

# ====== 是否做 VC（二选一）=====
# 0：只用 Stage1 instruct2（更少伪影、情绪更完整）
# 1：Stage1 + VC（音色更“保险”，但可能轻微压平情绪）
DO_VC="${DO_VC:-0}"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT_DIR/outputs/runs/${TS}_zh_emotion_story_60s"
mkdir -p "$RUN_ROOT/inputs"

echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] REF_WAV=$REF_WAV"
echo "[INFO] DO_VC=$DO_VC GAP_MS=$GAP_MS FADE_MS=$FADE_MS"

JSONL="$RUN_ROOT/inputs/story.jsonl"
python tools/build_zh_emotion_story_jsonl.py \
  --ref_wav "$REF_WAV" \
  --out_jsonl "$JSONL" \
  --key_prefix "story"

# ===== Stage1: CosyVoice3 instruct2 =====
STAGE1_OUT="$RUN_ROOT/stage1/pred_audio"
mkdir -p "$STAGE1_OUT"

echo "[INFO] ===== Stage1: CosyVoice3 instruct2 (ZH + emotion) ====="
python -u tools/cosyvoice3_batch.py \
  --cosyvoice_repo "$COSYVOICE_REPO" \
  --model_dir "$COSYVOICE3_MODEL" \
  instruct2 \
  --jsonl "$JSONL" \
  --prompt_wav "$REF_WAV" \
  --out_dir "$STAGE1_OUT" \
  2>&1 | tee "$RUN_ROOT/stage1/instruct.log"

FINAL_DIR="$STAGE1_OUT"

# ===== Stage2: VC（可选）=====
if [[ "$DO_VC" == "1" ]]; then
  echo "[INFO] ===== Stage2: CosyVoice3 VC (timbre refine) ====="
  VC_OUT="$RUN_ROOT/stage2/pred_audio_vc"
  mkdir -p "$VC_OUT"
  python -u tools/cosyvoice3_batch.py \
    --cosyvoice_repo "$COSYVOICE_REPO" \
    --model_dir "$COSYVOICE3_MODEL" \
    vc \
    --src_dir "$STAGE1_OUT" \
    --prompt_wav "$REF_WAV" \
    --out_dir "$VC_OUT" \
    2>&1 | tee "$RUN_ROOT/stage2/vc.log"
  FINAL_DIR="$VC_OUT"
fi

# ===== 拼接成 1 分钟左右 wav =====
FINAL_WAV="$RUN_ROOT/final_zh_emotion_story.wav"
python tools/concat_wavs_v3.py \
  --in_dir "$FINAL_DIR" \
  --out_wav "$FINAL_WAV" \
  --pattern "story_*.wav" \
  --gap_ms "$GAP_MS" \
  --fade_ms "$FADE_MS"

echo "[DONE] FINAL_WAV=$FINAL_WAV"
echo "[DONE] RUN_ROOT=$RUN_ROOT"