#!/usr/bin/env bash
# Path B (classic): HF -> TensorRT-LLM checkpoint -> trtllm-build -> engine -> run
# Run from a cloned TensorRT-LLM repo (for examples/) inside the TensorRT-LLM container.
set -euo pipefail
MODEL=${MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}
TRTLLM=${TRTLLM:-/opt/TensorRT-LLM}   # path to TensorRT-LLM source checkout
huggingface-cli download "$MODEL" --local-dir hf_models/tinyllama

# 1) HF weights -> TensorRT-LLM checkpoint (config.json + rank0.safetensors)
python "$TRTLLM/examples/models/core/llama/convert_checkpoint.py" \
  --model_dir hf_models/tinyllama --output_dir ckpt/tinyllama_fp16 --dtype float16

# 2) Build: shapes and GPU arch are fixed here (heavy, done once)
trtllm-build --checkpoint_dir ckpt/tinyllama_fp16 --output_dir engines/tinyllama_fp16 \
  --gemm_plugin auto --max_batch_size 8 --max_input_len 512 --max_seq_len 1024

# 3) Run (fast, repeated)
python "$TRTLLM/examples/run.py" --engine_dir engines/tinyllama_fp16 \
  --tokenizer_dir hf_models/tinyllama --max_output_len 64 --input_text "What is a KV cache?"
