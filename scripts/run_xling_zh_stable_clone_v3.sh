#!/usr/bin/env bash
set -Eeuo pipefail
trap 'echo "[ERR] line=$LINENO cmd=$BASH_COMMAND exit=$?"; exit 1' ERR

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TEXT=""
TEXT_FILE=""
EMOTION="neutral"
TAG="v3"

usage () {
  cat <<EOU
Usage:
  bash scripts/run_xling_zh_stable_clone_v3.sh --text "<zh_text>" --emotion <angry|happy|sad|neutral> --tag <run_tag>
  OR
  bash scripts/run_xling_zh_stable_clone_v3.sh --text_file /path/to/text.txt --emotion ... --tag ...

Env:
  STAGE1_PROMPT_WAV=/data/lizhen/EmoVoice/data/speakers/lizhen_ch_neutral.wav
  SKIP_VC=0/1
  NO_SPLIT=0/1
  KEEP_EXACT=0/1   # 1: 不做标点 normalize，尽量保持原文
  MAX_CHARS=36 MIN_CHARS=14 GAP_MS=120 FADE_MS=10
EOU
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --text) TEXT="$2"; shift 2 ;;
    --text_file) TEXT_FILE="$2"; shift 2 ;;
    --emotion) EMOTION="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERR] unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$TEXT" && -z "$TEXT_FILE" ]]; then
  echo "[ERR] need --text or --text_file"; exit 2
fi

REF_WAV="/data/lizhen/EmoVoice/data/speakers/ljh_neutral_22k.wav"
COSYVOICE_REPO="${COSYVOICE_REPO:-$ROOT_DIR/third_party/CosyVoice}"
COSYVOICE3_MODEL="${COSYVOICE3_MODEL:-$ROOT_DIR/ckpts/CosyVoice/Fun-CosyVoice3.0}"

MAX_CHARS="${MAX_CHARS:-36}"
MIN_CHARS="${MIN_CHARS:-14}"
GAP_MS="${GAP_MS:-120}"
FADE_MS="${FADE_MS:-10}"
SKIP_VC="${SKIP_VC:-0}"
NO_SPLIT="${NO_SPLIT:-0}"
KEEP_EXACT="${KEEP_EXACT:-1}"

STAGE1_PROMPT_WAV="${STAGE1_PROMPT_WAV:-/data/lizhen/EmoVoice/data/speakers/lizhen_ch_neutral.wav}"
if [[ ! -f "${STAGE1_PROMPT_WAV}" ]]; then
  echo "[WARN] STAGE1_PROMPT_WAV not found: ${STAGE1_PROMPT_WAV}"
  STAGE1_PROMPT_WAV="${COSYVOICE_REPO}/asset/cross_lingual_prompt.wav"
fi

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$ROOT_DIR/outputs/runs/${TS}_xling_zh_${TAG}_${EMOTION}"
mkdir -p "$RUN_ROOT/inputs"

# 永远落地一个 text.txt，避免命令行参数被搞乱
TEXT_TXT="$RUN_ROOT/inputs/text.txt"
if [[ -n "$TEXT_FILE" ]]; then
  cp -f "$TEXT_FILE" "$TEXT_TXT"
else
  # printf 比 echo 更稳，不会吞反斜杠/不一致换行
  printf "%s" "$TEXT" > "$TEXT_TXT"
fi

echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] TEXT_TXT=$TEXT_TXT"
echo "[INFO] REF_WAV=$REF_WAV"
echo "[INFO] STAGE1_PROMPT_WAV=$STAGE1_PROMPT_WAV"
echo "[INFO] MAX_CHARS=$MAX_CHARS MIN_CHARS=$MIN_CHARS GAP_MS=$GAP_MS FADE_MS=$FADE_MS SKIP_VC=$SKIP_VC NO_SPLIT=$NO_SPLIT KEEP_EXACT=$KEEP_EXACT"
echo "[INFO] TEXT=$(cat "$TEXT_TXT")"

FIXED_JSONL="$RUN_ROOT/inputs/fixed_v3.jsonl"

extra=()
if [[ "$NO_SPLIT" == "1" ]]; then extra+=(--no_split); fi
if [[ "$KEEP_EXACT" == "1" ]]; then extra+=(--keep_exact); fi

python tools/build_xling_fixed_jsonl_v3.py \
  --text_file "$TEXT_TXT" \
  --emotion "$EMOTION" \
  --ref_wav "$REF_WAV" \
  --out_fixed "$FIXED_JSONL" \
  --max_chars "$MAX_CHARS" \
  --min_chars "$MIN_CHARS" \
  --key_prefix "xling" \
  "${extra[@]}"

# 强校验：JSONL 拼回去必须等于 text.txt
python - "$FIXED_JSONL" "$TEXT_TXT" <<'PY'
import json, re, sys
jpath, tpath = sys.argv[1], sys.argv[2]

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", "", s)
    return s

orig = open(tpath, "r", encoding="utf-8").read().strip()
segs = []
with open(jpath, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            segs.append(json.loads(line)["source_text"])
joined = "".join(segs)

print("[CHECK] orig  =", orig)
print("[CHECK] joined=", joined)

if norm(orig) != norm(joined):
    raise SystemExit("[ERR] fixed_v3.jsonl does not match text.txt (input corrupted)")
print("[CHECK] OK: fixed jsonl matches input text.")
PY

echo "[INFO] ===== Stage1: CosyVoice3 instruct2 ====="
STAGE1_OUT="$RUN_ROOT/stage1_cv3/pred_audio"
mkdir -p "$STAGE1_OUT"
python -u tools/cosyvoice3_batch.py \
  --cosyvoice_repo "$COSYVOICE_REPO" \
  --model_dir "$COSYVOICE3_MODEL" \
  instruct2 \
  --jsonl "$FIXED_JSONL" \
  --prompt_wav "$STAGE1_PROMPT_WAV" \
  --out_dir "$STAGE1_OUT" \
  --concat 2>&1 | tee "$RUN_ROOT/stage1_cv3/instruct.log"

if [[ "$SKIP_VC" == "1" ]]; then
  FINAL_WAV="$RUN_ROOT/final_stage1_only.wav"
  python tools/concat_wavs_v3.py --in_dir "$STAGE1_OUT" --out_wav "$FINAL_WAV" --pattern "xling_*.wav" --gap_ms "$GAP_MS" --fade_ms "$FADE_MS"
  echo "[DONE] RUN_ROOT=$RUN_ROOT"
  echo "[DONE] FINAL_WAV=$FINAL_WAV"
  exit 0
fi

echo "[INFO] ===== Stage2: CosyVoice3 VC ====="
VC_OUT="$RUN_ROOT/stage2_vc/pred_audio_vc"
mkdir -p "$VC_OUT"
python -u tools/cosyvoice3_batch.py \
  --cosyvoice_repo "$COSYVOICE_REPO" \
  --model_dir "$COSYVOICE3_MODEL" \
  vc \
  --src_dir "$STAGE1_OUT" \
  --prompt_wav "$REF_WAV" \
  --out_dir "$VC_OUT" \
  --concat 2>&1 | tee "$RUN_ROOT/stage2_vc/vc.log"

FINAL_WAV="$RUN_ROOT/final_zh_v3.wav"
python tools/concat_wavs_v3.py --in_dir "$VC_OUT" --out_wav "$FINAL_WAV" --pattern "xling_*.wav" --gap_ms "$GAP_MS" --fade_ms "$FADE_MS"
echo "[DONE] RUN_ROOT=$RUN_ROOT"
echo "[DONE] FINAL_WAV=$FINAL_WAV"
