#!/usr/bin/env bash
set -Eeuo pipefail

TEXT=""
TAG="ljh"

usage () {
  echo 'Usage: bash scripts/run_xling_zh_stable_clone_v3_4emo.sh --text "<zh_text>" --tag <tag_prefix>'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --text) TEXT="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERR] unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$TEXT" ]]; then echo "[ERR] --text required"; exit 2; fi

TS="$(date +%Y%m%d_%H%M%S)"
BATCH_DIR="outputs/runs/${TS}_xling_zh_${TAG}_4emo"
mkdir -p "${BATCH_DIR}"
echo -e "emotion\tfinal_wav\trun_root" > "${BATCH_DIR}/summary.tsv"

for emo in happy angry sad neutral; do
  echo "[INFO] ===== ${emo} ====="
  LOG="${BATCH_DIR}/${emo}.log"

  bash scripts/run_xling_zh_stable_clone_v3.sh \
    --text "${TEXT}" \
    --emotion "${emo}" \
    --tag "${TAG}" | tee "${LOG}"

  FINAL_WAV="$(grep -m1 '\[DONE\] FINAL_WAV=' "${LOG}" | sed 's/.*FINAL_WAV=//')"
  RUN_ROOT="$(grep -m1 '\[DONE\] RUN_ROOT=' "${LOG}" | sed 's/.*RUN_ROOT=//')"

  echo -e "${emo}\t${FINAL_WAV}\t${RUN_ROOT}" >> "${BATCH_DIR}/summary.tsv"
  echo "[INFO] ${emo} -> ${FINAL_WAV}"
done

echo "[DONE] batch_dir=${BATCH_DIR}"
echo "[DONE] summary=${BATCH_DIR}/summary.tsv"
