# NKI
MEng Project for 25fall

*Group Member: Jifeng Wu, Kevin Golmes, Shiqi Liu, Chia Liu*

## Setup Instructions

1. Activate the Neuron Distributed Inference environment
```Shell
source /opt/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate
```

2. Download gpt-oss-20b from HuggingFace
```Shell
hf download openai/gpt-oss-20b --local-dir ~/models/gpt-oss-20b/
```

3. Run main.py
```
python main.py
```