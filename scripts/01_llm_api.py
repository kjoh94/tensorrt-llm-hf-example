"""Path A (default LLM API / PyTorch backend): HF model id -> TensorRT-LLM runtime.
No separate engine-build step; weights are loaded from the Hub.
"""
import argparse
from tensorrt_llm import LLM, SamplingParams

p = argparse.ArgumentParser()
p.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
p.add_argument("--max_tokens", type=int, default=64)
a = p.parse_args()

prompts = ["What is a KV cache?", "Explain prefill vs decode in one sentence."]
llm = LLM(model=a.model)  # loads HF config + weights, sets up paged KV cache
outs = llm.generate(prompts, SamplingParams(max_tokens=a.max_tokens, temperature=0.0))
for o in outs:
    print("PROMPT:", o.prompt)
    print("OUTPUT:", o.outputs[0].text, "\n")
