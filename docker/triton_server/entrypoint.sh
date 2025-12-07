#!/bin/bash
set -euo pipefail

# Простая проверка наличия "реального" или "симулированного" GPU
has_nvidia_smi() {
  command -v nvidia-smi >/dev/null 2>&1 || return 1
  # nvidia-smi может вернуть non-zero если драйвер отсутствует — проверим список GPU
  nvidia-smi -L >/dev/null 2>&1
}

echo "[$(date -Iseconds)] ENTRYPOINT: checking GPU presence..."
if has_nvidia_smi; then
  echo "[$(date -Iseconds)] GPU detected (nvidia-smi available). Starting Triton in GPU mode."
else
  echo "[$(date -Iseconds)] No GPU detected. Starting Triton in CPU-only mode (fallback)."
fi

# Вы можете добавить дополнительные флаги в TRITON_ARGS через переменную окружения
TRITON_ARGS="${TRITON_ARGS:-}"

echo "[$(date -Iseconds)] Launching: tritonserver --model-repository=/models $TRITON_ARGS"
exec tritonserver --model-repository=/models $TRITON_ARGS
