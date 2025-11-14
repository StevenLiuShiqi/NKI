import os
import sys

# Add the NKI project directory to Python path so src.model can be imported
sys.path.insert(0, "/home/ubuntu/NKI")

os.environ["VLLM_NEURON_FRAMEWORK"] = "neuronx-distributed-inference"

from vllm import LLM, SamplingParams

# Create an LLM.
llm = LLM(
   model=os.path.expanduser("~/models/gpt-oss-20b/"),
   tensor_parallel_size=32,
   max_num_seqs=1,
   max_model_len=16384,
   device="neuron",
   use_v2_block_manager=True,
   override_neuron_config={},
)

# Sample prompts.
prompts = [
   "The meaning of life is",
]
outputs = llm.generate(prompts, SamplingParams(top_k=1))

for output in outputs:
   prompt = output.prompt
   generated_text = output.outputs[0].text
   print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")