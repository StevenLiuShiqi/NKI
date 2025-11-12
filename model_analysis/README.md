# GPT-OSS-20B Model Analysis

## Files Overview

### 1. `extract_state_dicts_and_inputs_outputs.py`

A Python script that extracts model weights and records input/output data for each layer during inference:

- `*.pt` files: PyTorch state dictionaries for each layer
- `*.json` files: Input/output recordings for verification

### 2. Analysis Notebook

A Jupyter notebook containing implementations and tests for various model components:
- `GptOssTopKRouter` - Top-k routing mechanism
- `GptOssRMSNorm` - RMS normalization layer
- `GptOssRotaryEmbedding` - Rotary position embeddings
- `GptOssExperts` - Mixture of Experts implementation
- `GptOssAttention` - Self-attention mechanism
- `GptOssMLP` - Multi-layer perceptron with routing
- `GptOssDecoderLayer` - Complete transformer decoder layer

## Prerequisites

Before running the analysis notebook, you must first execute the extraction script:

```bash
python extract_state_dicts_and_inputs_outputs.py
```

This will:
- Load the GPT-OSS-20B model from `~/models/gpt-oss-20b/`
- Extract state dictionaries (weights) for each layer and save as `.pt` files
- Record inputs and outputs during model inference and save as `.json` files
- Generate test data using the prompt: "I believe the meaning of life is"
