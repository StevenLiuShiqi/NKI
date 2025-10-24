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

Ignore this error message:

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
neuronx-distributed-inference 0.6.10598+a59fdc00 requires transformers==4.51.*, but you have transformers 4.57.1 which is incompatible.
```

Run main.py:

```Shell
python main.py
```

Notes:

- The first time you run this script, `import torch` take an awfully long time (~5 minutes).
- Compiling and saving the model takes ~5-6 minutes.
