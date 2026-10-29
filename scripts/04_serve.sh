#!/usr/bin/env bash
# OpenAI-compatible server on top of the same runtime.
set -euo pipefail
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
trtllm-serve "$MODEL" --port 8000 &
sleep 60
curl -s localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL\",\"prompt\":\"What is a KV cache?\",\"max_tokens\":32}"
