# NKI
MEng Project for 25fall

*Group Member: Jifeng Wu, Kevin Golmes, Shiqi Liu, Chia Liu*

## Setup Instructions

Requires a trn1.32xlarge instance.
1. Activate the Neuron Distributed Inference environment
```Shell
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
```

2. Download Unsloth's bf16 version of gpt-oss-20b from HuggingFace
```Shell
hf download unsloth/gpt-oss-20b-BF16 --local-dir ~/models/gpt-oss-20b/
```

3. Replace `hf_adapter.py` in environment
```
cp new_hf_adapter.py /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/utils/hf_adapter.py/opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/utils/hf_adapter.py
```

4. Run main.py
```
python main.py
```