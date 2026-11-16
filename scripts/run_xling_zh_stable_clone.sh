set -Eeuo pipefail
trap 'echo "[ERR] line=$LINENO cmd=$BASH_COMMAND exit=$?"; exit 1' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# ---------- args ----------
ZH_TEXT=""
EMOTION="neutral"
RUN_TAG="stable"

# MODE options:
#   cv3_stage1_vc   : Stage1=CosyVoice3 instruct2 (generic prompt) -> Stage2=VC (ljh)
#   cv3_direct      : CosyVoice3 instruct2 directly with ljh prompt (no VC)
#   emovoice_pp_vc  : Stage1=EmoVoice-PP -> Stage2=VC (你现在这条)
MODE="${MODE:-cv3_stage1_vc}"

usage () {
  cat <<EOU
Usage:
  bash scripts/run_xling_zh_stable_clone.sh --text "<zh_text>" --emotion <angry|happy|sad|neutral> [--tag xxx]

Env:
  MODE=cv3_stage1_vc | cv3_direct | emovoice_pp_vc
  MAX_CHARS=22
  MAX_NEW_TOKENS=8000                # only used by emovoice_pp_vc
  STAGE1_PROMPT_WAV=...              # only used by cv3_stage1_vc
EOU
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --text) ZH_TEXT="$2"; shift 2 ;;
    --emotion) EMOTION="$2"; shift 2 ;;
    --tag) RUN_TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERR] unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$ZH_TEXT" ]]; then echo "[ERR] --text is required"; usage; exit 1; fi

# ---------- fixed target speaker ----------
REF_WAV="/data/lizhen/EmoVoice/data/speakers/ljh_neutral_22k.wav"

# ---------- knobs ----------
MAX_CHARS="${MAX_CHARS:-22}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8000}"

# ---------- ckpts / paths ----------
LLM_PATH="${LLM_PATH:-$ROOT_DIR/ckpts/Qwen/Qwen2.5-0.5B}"
PHN_TOKENIZER="${PHN_TOKENIZER:-$ROOT_DIR/ckpts/Qwen/Qwen2.5-0.5B-phn}"
CODEC_PATH="${CODEC_PATH:-$ROOT_DIR/ckpts/CosyVoice/CosyVoice-300M-SFT}"
CKPT_PP="${CKPT_PP:-$ROOT_DIR/ckpts/EmoVoice-PP/EmoVoice-PP.pt}"

COSYVOICE_REPO="${COSYVOICE_REPO:-$ROOT_DIR/third_party/CosyVoice}"
COSYVOICE3_MODEL="${COSYVOICE3_MODEL:-$ROOT_DIR/ckpts/CosyVoice/Fun-CosyVoice3.0}"

# Stage1 generic prompt wav (for cv3_stage1_vc)
STAGE1_PROMPT_WAV="${STAGE1_PROMPT_WAV:-$COSYVOICE_REPO/asset/cross_lingual_prompt.wav}"
if [[ "${MODE}" == "cv3_stage1_vc" ]]; then
  if [[ ! -f "${STAGE1_PROMPT_WAV}" ]]; then
    # fallback
    STAGE1_PROMPT_WAV="$COSYVOICE_REPO/asset/zero_shot_prompt.wav"
  fi
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT_DIR}/outputs/runs/${TS}_xling_zh_${RUN_TAG}"
mkdir -p "${RUN_ROOT}/inputs" "${RUN_ROOT}/metrics"
RUN_LOG="${RUN_ROOT}/run.log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] MODE=${MODE} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"(not set)"}"
echo "[INFO] REF_WAV=${REF_WAV}"
echo "[INFO] MAX_CHARS=${MAX_CHARS} MAX_NEW_TOKENS=${MAX_NEW_TOKENS}"
if [[ "${MODE}" == "cv3_stage1_vc" ]]; then
  echo "[INFO] STAGE1_PROMPT_WAV=${STAGE1_PROMPT_WAV}"
fi

python - <<PY
import os
need = [
  "${REF_WAV}",
  "${COSYVOICE3_MODEL}",
  "${COSYVOICE3_MODEL}/campplus.onnx",
]
missing=[p for p in need if not os.path.exists(p)]
print("[CHECK] missing:", missing)
if missing:
  raise SystemExit(2)
PY

# ---------- build fixed jsonl (split long zh into short segments) ----------
FIXED_JSONL="${RUN_ROOT}/inputs/fixed.jsonl"
python tools/build_xling_fixed_jsonl.py \
  --text "${ZH_TEXT}" \
  --emotion "${EMOTION}" \
  --ref_wav "${REF_WAV}" \
  --out_fixed "${FIXED_JSONL}" \
  --max_chars "${MAX_CHARS}" \
  --key_prefix "xling"

FINAL_AUDIO_DIR=""

# =========================================================
# MODE 1: cv3_stage1_vc  (RECOMMENDED for Chinese stability)
# =========================================================
if [[ "${MODE}" == "cv3_stage1_vc" ]]; then
  echo "[INFO] ===== Stage1: CosyVoice3 instruct2 (generic voice, make sure it's Mandarin) ====="
  STAGE1_OUT="${RUN_ROOT}/stage1_cv3/pred_audio"
  mkdir -p "${STAGE1_OUT}"
  python -u tools/cosyvoice3_batch.py \
    --cosyvoice_repo "${COSYVOICE_REPO}" \
    --model_dir "${COSYVOICE3_MODEL}" \
    instruct2 \
    --jsonl "${FIXED_JSONL}" \
    --prompt_wav "${STAGE1_PROMPT_WAV}" \
    --out_dir "${STAGE1_OUT}" \
    --concat 2>&1 | tee "${RUN_ROOT}/stage1_cv3/instruct.log"

  echo "[INFO] ===== Stage2: CosyVoice3 VC (transfer to ljh) ====="
  VC_OUT="${RUN_ROOT}/stage2_vc/pred_audio_vc"
  mkdir -p "${VC_OUT}"
  python -u tools/cosyvoice3_batch.py \
    --cosyvoice_repo "${COSYVOICE_REPO}" \
    --model_dir "${COSYVOICE3_MODEL}" \
    vc \
    --src_dir "${STAGE1_OUT}" \
    --prompt_wav "${REF_WAV}" \
    --out_dir "${VC_OUT}" \
    --concat 2>&1 | tee "${RUN_ROOT}/stage2_vc/vc.log"

  FINAL_AUDIO_DIR="${VC_OUT}"
fi

# ==========================================
# MODE 2: cv3_direct (one-step in ljh timbre)
# ==========================================
if [[ "${MODE}" == "cv3_direct" ]]; then
  echo "[INFO] ===== CosyVoice3 instruct2 directly with ljh prompt (no VC) ====="
  OUT_DIR="${RUN_ROOT}/cv3_direct/pred_audio"
  mkdir -p "${OUT_DIR}"
  python -u tools/cosyvoice3_batch.py \
    --cosyvoice_repo "${COSYVOICE_REPO}" \
    --model_dir "${COSYVOICE3_MODEL}" \
    instruct2 \
    --jsonl "${FIXED_JSONL}" \
    --prompt_wav "${REF_WAV}" \
    --out_dir "${OUT_DIR}" \
    --concat 2>&1 | tee "${RUN_ROOT}/cv3_direct/instruct.log"
  FINAL_AUDIO_DIR="${OUT_DIR}"
fi

# ==========================================
# MODE 3: emovoice_pp_vc (your current chain)
# ==========================================
if [[ "${MODE}" == "emovoice_pp_vc" ]]; then
  echo "[INFO] ===== Stage1: EmoVoice-PP (ZH + emotion) ====="
  STAGE1_OUT="${RUN_ROOT}/emovoicePP_zh"
  mkdir -p "${STAGE1_OUT}"

  # stable decode params
  DO_SAMPLE=false
  TEMPERATURE=1.0
  TOP_P=1.0
  TOP_K=0
  TEXT_RP=1.2
  AUDIO_RP=1.2
  NO_REPEAT_NGRAM=0

  CODE_LAYER=3
  TOTAL_AUDIO_VOCABSIZE=4160
  TOTAL_VOCABSIZE=156160
  SPEECH_SR=22050

  python examples/tts/inference_tts.py \
    "hydra.run.dir=${STAGE1_OUT}" \
    "++model_config.llm_name=qwen2.5-0.5b" \
    "++model_config.llm_path=${LLM_PATH}" \
    "++model_config.llm_dim=896" \
    "++model_config.codec_decoder_path=${CODEC_PATH}" \
    "++model_config.codec_decode=true" \
    "++model_config.codec_decoder_type=CosyVoice" \
    "++model_config.group_decode=true" \
    "++model_config.group_decode_adapter_type=linear" \
    "++model_config.use_text_stream=false" \
    "++model_config.vocab_config.code_layer=${CODE_LAYER}" \
    "++model_config.vocab_config.total_audio_vocabsize=${TOTAL_AUDIO_VOCABSIZE}" \
    "++model_config.vocab_config.total_vocabsize=${TOTAL_VOCABSIZE}" \
    "++model_config.phn_tokenizer=${PHN_TOKENIZER}" \
    "++dataset_config.dataset=speech_dataset_tts" \
    "++dataset_config.val_data_path=${FIXED_JSONL}" \
    "++dataset_config.train_data_path=${FIXED_JSONL}" \
    "++dataset_config.inference_mode=true" \
    "++dataset_config.use_emo=true" \
    "++dataset_config.use_text_stream=false" \
    "++dataset_config.num_latency_tokens=5" \
    "++dataset_config.do_layershift=false" \
    "++dataset_config.vocab_config.code_layer=${CODE_LAYER}" \
    "++dataset_config.vocab_config.total_audio_vocabsize=${TOTAL_AUDIO_VOCABSIZE}" \
    "++dataset_config.vocab_config.total_vocabsize=${TOTAL_VOCABSIZE}" \
    "++train_config.model_name=tts" \
    "++train_config.freeze_encoder=true" \
    "++train_config.freeze_llm=true" \
    "++train_config.freeze_group_decode_adapter=true" \
    "++train_config.batching_strategy=custom" \
    "++train_config.num_epochs=1" \
    "++train_config.val_batch_size=1" \
    "++train_config.num_workers_dataloader=0" \
    "++decode_config.max_new_tokens=${MAX_NEW_TOKENS}" \
    "++decode_config.do_sample=${DO_SAMPLE}" \
    "++decode_config.temperature=${TEMPERATURE}" \
    "++decode_config.top_p=${TOP_P}" \
    "++decode_config.top_k=${TOP_K}" \
    "++decode_config.text_repetition_penalty=${TEXT_RP}" \
    "++decode_config.audio_repetition_penalty=${AUDIO_RP}" \
    "++decode_config.no_repeat_ngram_size=${NO_REPEAT_NGRAM}" \
    "++decode_config.decode_text_only=false" \
    "++decode_config.do_layershift=false" \
    "++decode_config.input_text=true" \
    "++decode_config.num_latency_tokens=5" \
    "++speech_sample_rate=${SPEECH_SR}" \
    "++decode_log=${STAGE1_OUT}" \
    "++ckpt_path=${CKPT_PP}" \
    "++log_config.log_file=${STAGE1_OUT}/infer.log"

  echo "[INFO] ===== Stage2: CosyVoice3 VC ====="
  SRC_DIR="${STAGE1_OUT}/pred_audio/neutral_prompt_speech"
  VC_OUT="${STAGE1_OUT}/pred_audio_vc/neutral_prompt_speech"
  python -u tools/cosyvoice3_batch.py \
    --cosyvoice_repo "${COSYVOICE_REPO}" \
    --model_dir "${COSYVOICE3_MODEL}" \
    vc \
    --src_dir "${SRC_DIR}" \
    --prompt_wav "${REF_WAV}" \
    --out_dir "${VC_OUT}" \
    --concat 2>&1 | tee "${STAGE1_OUT}/vc.log"

  FINAL_AUDIO_DIR="${VC_OUT}"
fi

if [[ -z "${FINAL_AUDIO_DIR}" ]]; then
  echo "[ERR] FINAL_AUDIO_DIR empty. MODE=${MODE} not recognized?"
  exit 2
fi

FINAL_WAV="${RUN_ROOT}/final_zh.wav"
python tools/concat_wavs.py --in_dir "${FINAL_AUDIO_DIR}" --out_wav "${FINAL_WAV}" --pattern "xling_*.wav"

echo "[DONE] RUN_ROOT=${RUN_ROOT}"
echo "[DONE] FINAL_WAV=${FINAL_WAV}"
