#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERR] line=$LINENO cmd=$BASH_COMMAND exit=$?"; exit 1' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JSONL_IN="${1:-}"
if [[ -z "${JSONL_IN}" ]]; then
  echo "Usage: bash scripts/run_lizhen_en_compare_Aplus.sh <path/to.jsonl>"
  exit 1
fi
JSONL_IN="$(realpath "${JSONL_IN}")"

# -------- knobs --------
PRESET="${PRESET:-strong}"         # strong / medium
CUDA="${CUDA:-0}"
NUM_WORKERS="${NUM_WORKERS:-0}"
NEUTRAL_WAV="${NEUTRAL_WAV:-/data/lizhen/EmoVoice/data/speakers/lizhen_neutral_22k.wav}"

# 关键：强制把 JSONL 里的 neutral_speaker_wav 统一替换为 NEUTRAL_WAV（默认开启）
FORCE_NEUTRAL_WAV="${FORCE_NEUTRAL_WAV:-1}"   # 1=force override; 0=only fill when missing

# 评测开关：默认只做 Whisper+WER（且 whisper 默认 CPU），emo_acc 默认跳过
RUN_WHISPER="${RUN_WHISPER:-1}"              # 1/0
WHISPER_DEVICE="${WHISPER_DEVICE:-cpu}"      # cpu/cuda
RUN_WER="${RUN_WER:-1}"                      # 1/0
RUN_EMO_ACC="${RUN_EMO_ACC:-0}"              # 1/0（需要 gt 里有 target_wav 才有意义）

SYN_TEXT="${SYN_TEXT:-Today I will demonstrate a reproducible emotional TTS pipeline. \
Angry, happy, and sad should be clearly distinguishable while pronunciation stays correct.}"

# -------- ckpt layout --------
LLM_PATH="${LLM_PATH:-${ROOT_DIR}/ckpts/Qwen/Qwen2.5-0.5B}"
PHN_TOKENIZER="${PHN_TOKENIZER:-${ROOT_DIR}/ckpts/Qwen/Qwen2.5-0.5B-phn}"
CODEC_PATH="${CODEC_PATH:-${ROOT_DIR}/ckpts/CosyVoice/CosyVoice-300M-SFT}"

CKPT_EV="${CKPT_EV:-${ROOT_DIR}/ckpts/EmoVoice/EmoVoice.pt}"
CKPT_PP="${CKPT_PP:-${ROOT_DIR}/ckpts/EmoVoice-PP/EmoVoice-PP.pt}"

# -------- vocab --------
CODE_LAYER=3
TOTAL_AUDIO_VOCABSIZE=4160
TOTAL_VOCABSIZE=156160
SPEECH_SR=22050

export CUDA_VISIBLE_DEVICES="${CUDA}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${ROOT_DIR}/outputs/runs/${TS}_lizhen_en_Aplus_${PRESET}"
mkdir -p "${RUN_ROOT}/inputs"
RUN_LOG="${RUN_ROOT}/run.log"

exec > >(tee -a "${RUN_LOG}") 2>&1

echo "[INFO] RUN_ROOT=${RUN_ROOT}"
echo "[INFO] JSONL_IN=${JSONL_IN}"
echo "[INFO] PRESET=${PRESET} CUDA=${CUDA} NUM_WORKERS=${NUM_WORKERS}"
echo "[INFO] NEUTRAL_WAV=${NEUTRAL_WAV}"
echo "[INFO] FORCE_NEUTRAL_WAV=${FORCE_NEUTRAL_WAV}"
echo "[INFO] SYN_TEXT=${SYN_TEXT}"
echo "[INFO] RUN_WHISPER=${RUN_WHISPER} WHISPER_DEVICE=${WHISPER_DEVICE} RUN_WER=${RUN_WER} RUN_EMO_ACC=${RUN_EMO_ACC}"

# -------- sanity checks --------
python - <<PY
import os, sys
need = [
  "${NEUTRAL_WAV}",
  "${LLM_PATH}/config.json",
  "${LLM_PATH}/tokenizer.json",
  "${CODEC_PATH}",
  "${CKPT_EV}",
  "${CKPT_PP}",
  "${PHN_TOKENIZER}",
]
missing=[p for p in need if not os.path.exists(p)]
print("[CHECK] missing:", missing)
if missing:
  raise SystemExit(2)
w=[x for x in os.listdir("${LLM_PATH}") if x.startswith("model") and (x.endswith(".safetensors") or x.endswith(".bin"))]
print("[CHECK] LLM weights:", w, "count=", len(w))
PY

# -------- build fixed jsonl --------
JSONL_FIXED="${RUN_ROOT}/inputs/$(basename "${JSONL_IN%.jsonl}").fixed.jsonl"
python - "${JSONL_IN}" "${JSONL_FIXED}" <<'PY'
import json, sys, os

inp, outp = sys.argv[1], sys.argv[2]
neutral = os.environ["NEUTRAL_WAV"]
synth = os.environ.get("SYN_TEXT","").strip()
force = os.environ.get("FORCE_NEUTRAL_WAV","1").strip() != "0"

def norm_emo(e: str) -> str:
    e=(e or "neutral").strip().lower()
    alias={"joy":"happy","happiness":"happy","anger":"angry","sadness":"sad","sorrow":"sad"}
    return alias.get(e, e)

def strong_prompt(emotion: str) -> str:
    emo = (emotion or "neutral").lower()
    table = {
      "angry":  "Speak with controlled anger: higher energy, sharper attacks, slightly faster pace, tense voice, short pauses.",
      "happy":  "Speak with genuine happiness: brighter tone, higher pitch, smiling voice, lively rhythm, slightly faster pace.",
      "sad":    "Speak with sadness: lower energy, softer volume, slower pace, falling intonation, longer pauses.",
      "neutral":"Speak in a calm, neutral tone: steady pace, moderate pitch, clear articulation.",
    }
    base = table.get(emo, table["neutral"])
    return base + " Keep pronunciation clear and natural. Do NOT exaggerate into shouting."

count=0
with open(inp,"r",encoding="utf-8") as fin, open(outp,"w",encoding="utf-8") as fout:
    for i,line in enumerate(fin):
        line=line.strip()
        if not line:
            continue
        d=json.loads(line)

        emo = norm_emo(d.get("emotion"))
        d["emotion"] = emo

        # 强制改成本次要合成的文本（source_text/target_text）
        if synth:
            d["source_text"] = synth
            d["target_text"] = synth

        # 必需：source_len / target_len
        st = d.get("source_text","") or ""
        d["source_len"] = int(d.get("source_len") or max(1, len(st.split())))

        # 必需：answer_cosyvoice_speech_token 必须是 list（空 list 也行）
        if d.get("answer_cosyvoice_speech_token") is None or not isinstance(d.get("answer_cosyvoice_speech_token"), list):
            d["answer_cosyvoice_speech_token"] = []
        d["target_len"] = int(d.get("target_len") or len(d["answer_cosyvoice_speech_token"]))

        # speaker prompt：默认强制覆盖，确保真正用 NEUTRAL_WAV
        if force or not d.get("neutral_speaker_wav"):
            d["neutral_speaker_wav"] = neutral

        if not d.get("emotion_text_prompt"):
            d["emotion_text_prompt"] = strong_prompt(emo)

        if not d.get("key"):
            d["key"] = f"lizhen_{i:04d}_{emo}"

        fout.write(json.dumps(d, ensure_ascii=False) + "\n")
        count += 1

print(f"[INFO] wrote fixed jsonl: {outp} (n={count})")
PY

# -------- decode presets --------
# 参数含义参考 transformers generation 文档：top_k/top_p/temperature/do_sample :contentReference[oaicite:5]{index=5}
if [[ "${PRESET}" == "strong" ]]; then
  DO_SAMPLE=true
  TEMPERATURE=1.10
  TOP_P=0.90
  TOP_K=80
  TEXT_RP=1.35
  AUDIO_RP=1.15
  NO_REPEAT_NGRAM=3
  MAX_NEW_TOKENS=2500
else
  DO_SAMPLE=false
  TEMPERATURE=1.0
  TOP_P=1.0
  TOP_K=0
  TEXT_RP=1.2
  AUDIO_RP=1.2
  NO_REPEAT_NGRAM=0
  MAX_NEW_TOKENS=3000
fi

common_args=(
  "hydra.run.dir=."
  "++model_config.llm_name=qwen2.5-0.5b"
  "++model_config.llm_path=${LLM_PATH}"
  "++model_config.llm_dim=896"
  "++model_config.codec_decoder_path=${CODEC_PATH}"
  "++model_config.codec_decode=true"
  "++model_config.codec_decoder_type=CosyVoice"
  "++model_config.group_decode=true"
  "++model_config.group_decode_adapter_type=linear"
  "++model_config.use_text_stream=false"
  "++model_config.vocab_config.code_layer=${CODE_LAYER}"
  "++model_config.vocab_config.total_audio_vocabsize=${TOTAL_AUDIO_VOCABSIZE}"
  "++model_config.vocab_config.total_vocabsize=${TOTAL_VOCABSIZE}"

  "++dataset_config.dataset=speech_dataset_tts"
  "++dataset_config.val_data_path=${JSONL_FIXED}"
  "++dataset_config.train_data_path=${JSONL_FIXED}"
  "++dataset_config.inference_mode=true"
  "++dataset_config.use_emo=true"
  "++dataset_config.do_layershift=false"
  "++dataset_config.vocab_config.code_layer=${CODE_LAYER}"
  "++dataset_config.vocab_config.total_audio_vocabsize=${TOTAL_AUDIO_VOCABSIZE}"
  "++dataset_config.vocab_config.total_vocabsize=${TOTAL_VOCABSIZE}"

  "++train_config.model_name=tts"
  "++train_config.freeze_encoder=true"
  "++train_config.freeze_llm=true"
  "++train_config.freeze_group_decode_adapter=true"
  "++train_config.batching_strategy=custom"
  "++train_config.num_epochs=1"
  "++train_config.val_batch_size=1"
  "++train_config.num_workers_dataloader=${NUM_WORKERS}"

  "++decode_config.max_new_tokens=${MAX_NEW_TOKENS}"
  "++decode_config.do_sample=${DO_SAMPLE}"
  "++decode_config.temperature=${TEMPERATURE}"
  "++decode_config.top_p=${TOP_P}"
  "++decode_config.top_k=${TOP_K}"
  "++decode_config.text_repetition_penalty=${TEXT_RP}"
  "++decode_config.audio_repetition_penalty=${AUDIO_RP}"
  "++decode_config.no_repeat_ngram_size=${NO_REPEAT_NGRAM}"
  "++decode_config.decode_text_only=false"
  "++decode_config.do_layershift=false"
  # 关键：强制用输入文本合成（不要让 LLM 自己编文本）
  "++decode_config.input_text=true"
  "++speech_sample_rate=${SPEECH_SR}"
)

run_one() {
  local name="$1"
  local ckpt="$2"
  local num_latency="$3"
  shift 3
  local extra=("$@")

  local out="${RUN_ROOT}/${name}"
  mkdir -p "${out}"
  echo "[INFO] ===== RUN ${name} ====="
  echo "[INFO] out=${out}"

  python examples/tts/inference_tts.py \
    "hydra.run.dir=${out}" \
    "${common_args[@]}" \
    "++dataset_config.num_latency_tokens=${num_latency}" \
    "++decode_config.num_latency_tokens=${num_latency}" \
    "++decode_log=${out}" \
    "++ckpt_path=${ckpt}" \
    "++log_config.log_file=${out}/infer.log" \
    "${extra[@]}"

  # -------- optional evals (never block main inference) --------
  if [[ "${RUN_WHISPER}" == "1" ]]; then
    echo "[INFO] whisper decode (device=${WHISPER_DEVICE})"
    if [[ "${WHISPER_DEVICE}" == "cpu" ]]; then
      # 隐藏 GPU，避免 whisper 抢显存导致 OOM
      CUDA_VISIBLE_DEVICES="" python examples/tts/utils/decode_whisper.py \
        --parent_dir "${out}" --audio_subdir pred_audio/neutral_prompt_speech || true
    else
      python examples/tts/utils/decode_whisper.py \
        --parent_dir "${out}" --audio_subdir pred_audio/neutral_prompt_speech || true
    fi

    if [[ "${RUN_WER}" == "1" ]]; then
      if [[ -f "${out}/pred_whisper_text" ]]; then
        bash scripts/compute_wer.sh "${out}" || true
      else
        echo "[WARN] pred_whisper_text missing -> skip WER"
      fi
    fi
  fi

  if [[ "${RUN_EMO_ACC}" == "1" ]]; then
    if grep -q '"target_wav"' "${JSONL_FIXED}"; then
      python examples/tts/utils/eval_emo_acc.py --gt "${JSONL_FIXED}" --pred "${out}" --audio_subdir pred_audio/neutral_prompt_speech || true
    else
      echo "[WARN] gt jsonl has no target_wav -> skip eval_emo_acc.py"
    fi
  fi

  echo "[INFO] DONE ${name}"
  echo "[INFO] audio_dir=${out}/pred_audio/neutral_prompt_speech"
}

# EmoVoice：num_latency_tokens=0
run_one "emovoice_Aplus" "${CKPT_EV}" 0

# EmoVoice-PP：必须 phn_tokenizer + num_latency_tokens=5
run_one "emovoicePP_Aplus" "${CKPT_PP}" 5 \
  "++model_config.phn_tokenizer=${PHN_TOKENIZER}"

echo "[INFO] ALL DONE."
echo "[INFO] Compare:"
echo "  ${RUN_ROOT}/emovoice_Aplus/pred_audio/neutral_prompt_speech"
echo "  ${RUN_ROOT}/emovoicePP_Aplus/pred_audio/neutral_prompt_speech"
echo "[INFO] Logs:"
echo "  ${RUN_LOG}"
echo "[INFO] Fixed JSONL:"
echo "  ${JSONL_FIXED}"