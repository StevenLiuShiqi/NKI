import math
import torch

from neuronx_distributed_inference.utils.random import set_random_seed
from neuronx_distributed_inference.utils.testing import build_module, validate_accuracy
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear

# Module to test.
class ExampleModule(torch.nn.Module):
    def __init__(self, distributed, init_seed: int | None = None):
        super().__init__()
        if init_seed is not None:
            set_random_seed(init_seed)
        if distributed:
            self.linear = ColumnParallelLinear(
                input_size=2,
                output_size=2,
                bias=False,
                dtype=torch.float32,
            )
        else:
            self.linear = torch.nn.Linear(
                in_features=2,
                out_features=2,
                bias=False,
                dtype=torch.float32,
            )
        torch.nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x)


def test_validate_accuracy_basic_module():
    inputs = [(torch.arange(0, 2, dtype=torch.float32),)]
    example_inputs = [(torch.zeros((2), dtype=torch.float32),)]

    module_cpu = ExampleModule(distributed=False, init_seed=0)
    neuron_model = build_module(
        ExampleModule,
        example_inputs,
        module_init_kwargs={"distributed": True, "init_seed": 0},
    )

    def cpu_forward(x):
            return module_cpu(x)
    
    with torch.no_grad():
        expected_output = cpu_forward(*inputs[0])

    validate_accuracy(neuron_model, inputs, expected_outputs=[expected_output])

test_validate_accuracy_basic_module()
