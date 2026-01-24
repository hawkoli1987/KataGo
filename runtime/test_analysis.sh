#!/bin/bash
set -euo pipefail

ROOT_DIR="/scratch/Projects/SPEC-SF-AISG/source_files/KataGo"

INPUT_PATH="${1:-${ROOT_DIR}/runtime/assets/analysis/inputs/lategame_19.jsonl}"
OUTPUT_PATH="${2:-${ROOT_DIR}/runtime/assets/analysis/outputs/lategame_19.jsonl}"
HOST="${3:-hopper-34:9000}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

curl -s -X POST "http://${HOST}/analysis" --data-binary "@${INPUT_PATH}" > "${OUTPUT_PATH}"
echo "Saved output to ${OUTPUT_PATH}"
echo "Curl example:"
echo "curl -s -X POST http://${HOST}/analysis --data-binary @${INPUT_PATH}"
