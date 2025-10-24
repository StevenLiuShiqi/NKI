# NKI
MEng Project for 25fall

*Group Member: Jifeng Wu, Kevin Golmes, Shiqi Liu, Chia Liu*

## Setup Instructions

Requires a trn1.32xlarge instance. 

Activate the Neuron Distributed Inference environment:

```Shell
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
```

Download Unsloth's bf16 version of gpt-oss-20b from HuggingFace:

```Shell
hf download unsloth/gpt-oss-20b-BF16 --local-dir ~/models/gpt-oss-20b/
```

Replace `hf_adapter.py` in environment:

```Shell
cp new_hf_adapter.py /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/lib/python3.10/site-packages/neuronx_distributed_inference/utils/hf_adapter.py
```

Update Transformers:

```Shell
pip install --upgrade transformers
```

Run main.py:

```Shell
python main.py
```