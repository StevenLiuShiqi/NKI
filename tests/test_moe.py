import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import (
    GPTOSSInferenceConfig,
    NeuronGPTOSSConfig,
    # NeuronMLPBlock,
)
from src.gpt_oss import MLPBlock
from src.moe_classes import NeuronMLPBlock

from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy

from test_utils import _make_tiny_inference_config, _get_ref_config, _fill_module_parameters

_ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
_CHECKPOINT_PATH = _ARTIFACTS_DIR / "neuron_mlp_checkpoint.pt"
_CONSTANT_INIT_VALUE = 0.5


def test_validate_accuracy_basic_module():
    config = _make_tiny_inference_config()

    sample = torch.randn(12, config.hidden_size, dtype=config.neuron_config.torch_dtype)
    inputs = [(sample,)]
    example_inputs = [(torch.zeros_like(sample),)]

    reference_config = _get_ref_config(config=config)

    if _CHECKPOINT_PATH.exists():
        _CHECKPOINT_PATH.unlink()

    neuron_model = build_module(
        NeuronMLPBlock,
        example_inputs,
        tp_degree=1,
        module_init_kwargs={
            "config": config,
            "weight_init_value": _CONSTANT_INIT_VALUE,
        },
        checkpoint_path=str(_CHECKPOINT_PATH),
    )

    module_cpu = MLPBlock(
        config=reference_config,
        weight_init_value=_CONSTANT_INIT_VALUE,
    )

    _fill_module_parameters(module_cpu, _CONSTANT_INIT_VALUE)
    
    def cpu_forward(x):
        return module_cpu(x)
    
    with torch.no_grad():
        expected_output = cpu_forward(*inputs[0])


    validate_accuracy(neuron_model, inputs, expected_outputs=[expected_output])


test_validate_accuracy_basic_module()
