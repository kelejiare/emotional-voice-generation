#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

VAL_JSONL="$ROOT/data/infer/en_compare.jsonl"
COSYVOICE_DECODER="$ROOT/ckpts/CosyVoice/CosyVoice-300M-SFT"
LLM_05B="$ROOT/ckpts/Qwen/Qwen2.5-0.5B"
PHN_TOK="$ROOT/ckpts/Qwen/Qwen2.5-0.5B-phn"

CODE_LAYER=3
TOTAL_AUDIO_VOCAB=4160
TOTAL_VOCAB=156160
DO_LAYERSHIFT=false

RUN_ROOT="$ROOT/outputs/runs/$(date +%Y%m%d_%H%M%S)_en_compare_lizhen"
mkdir -p "$RUN_ROOT"
echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] VAL_JSONL=$VAL_JSONL"

run_one () {
  local OUT="$1"
  local CKPT="$2"
  local LAT="$3"
  shift 3
  local EXTRA=("$@")

  mkdir -p "$OUT"
  python examples/tts/inference_tts.py \
    hydra.run.dir="$OUT" \
    ++model_config.llm_name=qwen2.5-0.5b \
    ++model_config.llm_path="$LLM_05B" \
    ++model_config.llm_dim=896 \
    ++model_config.codec_decoder_path="$COSYVOICE_DECODER" \
    ++model_config.codec_decode=true \
    ++model_config.codec_decoder_type=CosyVoice \
    ++model_config.group_decode=true \
    ++model_config.group_decode_adapter_type=linear \
    ++model_config.use_text_stream=true \
    ++model_config.vocab_config.code_layer=$CODE_LAYER \
    ++model_config.vocab_config.total_audio_vocabsize=$TOTAL_AUDIO_VOCAB \
    ++model_config.vocab_config.total_vocabsize=$TOTAL_VOCAB \
    "${EXTRA[@]}" \
    ++dataset_config.dataset=speech_dataset_tts \
    ++dataset_config.val_data_path="$VAL_JSONL" \
    ++dataset_config.train_data_path="$VAL_JSONL" \
    ++dataset_config.inference_mode=true \
    ++dataset_config.use_emo=true \
    ++dataset_config.use_text_stream=false \
    ++dataset_config.num_latency_tokens=$LAT \
    ++dataset_config.do_layershift=$DO_LAYERSHIFT \
    ++dataset_config.vocab_config.code_layer=$CODE_LAYER \
    ++dataset_config.vocab_config.total_audio_vocabsize=$TOTAL_AUDIO_VOCAB \
    ++dataset_config.vocab_config.total_vocabsize=$TOTAL_VOCAB \
    ++train_config.model_name=tts \
    ++train_config.freeze_encoder=true \
    ++train_config.freeze_llm=true \
    ++train_config.freeze_group_decode_adapter=true \
    ++train_config.batching_strategy=custom \
    ++train_config.num_epochs=1 \
    ++train_config.val_batch_size=1 \
    ++train_config.num_workers_dataloader=2 \
    ++decode_config.max_new_tokens=3000 \
    ++decode_config.do_sample=false \
    ++decode_config.top_p=1.0 \
    ++decode_config.top_k=0 \
    ++decode_config.temperature=1.0 \
    ++decode_config.text_repetition_penalty=1.2 \
    ++decode_config.audio_repetition_penalty=1.2 \
    ++decode_config.decode_text_only=false \
    ++decode_config.num_latency_tokens=$LAT \
    ++decode_config.do_layershift=$DO_LAYERSHIFT \
    ++decode_log="$OUT" \
    ++ckpt_path="$CKPT" \
    ++speech_sample_rate=22050 \
    ++log_config.log_file="$OUT/infer.log"
}

# A) EmoVoice（LAT=0）
run_one "$RUN_ROOT/emovoice" "$ROOT/ckpts/EmoVoice/EmoVoice.pt" 0

# B) EmoVoice-PP（LAT=5 + phn_tokenizer）
run_one "$RUN_ROOT/emovoice_pp" "$ROOT/ckpts/EmoVoice-PP/EmoVoice-PP.pt" 5 \
  ++model_config.phn_tokenizer="$PHN_TOK"

echo "[DONE] Compare run_dir: $RUN_ROOT"
