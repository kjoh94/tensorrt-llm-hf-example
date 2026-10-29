# tensorrt-llm-hf-example

Minimal scripts for running a small public Hugging Face causal LM
(`TinyLlama/TinyLlama-1.1B-Chat-v1.0`) with NVIDIA TensorRT-LLM.
Companion to the blog post "TensorRT-LLM으로 Hugging Face 모델 돌리기" on https://kjoh94.github.io/.

## Two paths

| Path | Flow | When |
|---|---|---|
| A. LLM API / AutoDeploy | Hub id → Transformers config/weights → (AutoDeploy: `torch.export` → graph transforms) → TensorRT-LLM PyTorch runtime | Fastest way to try an HF model; no build step |
| B. Classic engine | HF → TRT-LLM checkpoint → `trtllm-build` → engine → runner | Fixed shapes/GPU, ahead-of-time engine |

```
scripts/01_llm_api.py        # Path A: LLM(model=hf_id).generate(...)
scripts/02_autodeploy.py     # Path A': AutoDeploy (beta) on an unmodified HF model
scripts/03_classic_engine.sh # Path B: convert_checkpoint -> trtllm-build -> run.py
scripts/04_serve.sh          # trtllm-serve OpenAI-compatible endpoint
```

## Environment (honest note)
These scripts need an NVIDIA GPU with a working TensorRT-LLM install — easiest is the
official TensorRT-LLM container from NGC, or a cloud GPU pod (e.g. RunPod) using it.
They were **not executed on a GPU in this repo's authoring environment**; commands follow the
official docs at the time of writing. TensorRT-LLM APIs (especially AutoDeploy, beta) change
between releases — check the version you install.

```bash
docker run --rm -it --gpus all --ipc=host -v $PWD:/work -w /work \
  nvcr.io/nvidia/tensorrt-llm/release:latest
python scripts/01_llm_api.py
```

Engines (`engines/`), checkpoints (`ckpt/`) and downloaded weights (`hf_models/`) are git-ignored.
An engine is tied to the GPU architecture, TensorRT-LLM version and max shapes it was built with.

## References
- https://huggingface.co/docs/transformers/main/en/community_integrations/tensorrt-llm
- https://nvidia.github.io/TensorRT-LLM/features/auto_deploy/auto-deploy.html
- https://nvidia.github.io/TensorRT-LLM/architecture/checkpoint.html
- https://nvidia.github.io/TensorRT-LLM/latest/features/kvcache.html
