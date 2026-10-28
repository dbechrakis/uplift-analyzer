#!/usr/bin/env bash
set -euo pipefail

MODEL_FILE="${MODEL_FILE:-Qwen3-4B-Instruct-2507-Q4_K_M.gguf}"
MODEL_URL="${MODEL_URL:-https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/${MODEL_FILE}}"
MODEL_SHA256="${MODEL_SHA256:-3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597}"
TARGET_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/models"

mkdir -p "${TARGET_DIR}"
if [[ ! -f "${TARGET_DIR}/${MODEL_FILE}" ]]; then
  curl --fail --location --output "${TARGET_DIR}/${MODEL_FILE}.part" "${MODEL_URL}"
  mv "${TARGET_DIR}/${MODEL_FILE}.part" "${TARGET_DIR}/${MODEL_FILE}"
fi
echo "${MODEL_SHA256}  ${TARGET_DIR}/${MODEL_FILE}" | sha256sum --check -
