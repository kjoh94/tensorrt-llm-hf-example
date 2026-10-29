"""Path A' (AutoDeploy, beta): HF Transformers model -> torch.export graph ->
automatic transforms (KV-cache insertion, attention kernel swap, GEMM fusion, sharding)
-> TensorRT-LLM PyTorch runtime (in-flight batching, paged KV cache).
The AutoDeploy API is beta and may change; see the official AutoDeploy docs.
"""
import argparse
from tensorrt_llm._torch.auto_deploy import LLM
from tensorrt_llm import SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
a = p.parse_args()

llm = LLM(model=a.model)
for o in llm.generate(["Hello, my name is"], SamplingParams(max_tokens=32)):
    print(o.outputs[0].text)
