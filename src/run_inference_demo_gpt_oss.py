#!/usr/bin/env python3
"""Run NxDI inference_demo with local GPT-OSS model class registration."""

import sys
from pathlib import Path

# Ensure local "src" package imports resolve when launched from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neuronx_distributed_inference import inference_demo
from src.gpt_oss.modeling_gpt_oss_orig import NeuronGptOssForCausalLM

import os
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"
os.environ["XLA_IR_DEBUG"]= "1"
os.environ["XLA_HLO_DEBUG"]= "1"
os.environ["NEURON_RT_INSPECT_ENABLE"]= "1"
os.environ["NEURON_RT_INSPECT_DEVICE_PROFILE"]= "1"
os.environ["NEURON_RT_INSPECT_OUTPUT_DIR"]= "./output"

def main():
    # Register local model implementation for the standard demo CLI.
    inference_demo.MODEL_TYPES["gpt_oss_orig"] = {"causal-lm": NeuronGptOssForCausalLM}
    # Optional: make gpt_oss resolve to local class too.
    inference_demo.MODEL_TYPES["gpt_oss"] = {"causal-lm": NeuronGptOssForCausalLM}
    # inference_demo.py currently parses --max-new-tokens but does not apply it.
    inference_demo.main()


if __name__ == "__main__":
    main()
