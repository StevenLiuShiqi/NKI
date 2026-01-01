import os.path
from glob import glob

import torch
from safetensors import safe_open
from safetensors_layer_grabber import yield_keys_and_tensors
from transformers import AutoTokenizer

from gpt_oss_simplified import GptOssForCausalLM, generate

DEVICE = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
MODEL_DIRECTORY_PATH = os.path.expanduser('~/models/gpt-oss-20b/')
SAFETENSORS_FILE_NAMES = glob(os.path.join(MODEL_DIRECTORY_PATH, '*.safetensors'))
SENTENCES = ['I believe the meaning of life is']


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIRECTORY_PATH)

    # Main code: assign tensors as you yield them
    print('Initializing model...')
    model = GptOssForCausalLM()       # Uninitialized, or on your desired device
    model.to(DEVICE)                  # Ensure model is already on correct device
    print('Model initialized.')

    print('Loading model weights from .safetensors files...')
    state_dict = model.state_dict()   # map of key->parameter/buffer (references, not clones)
    for key, tensor in yield_keys_and_tensors(SAFETENSORS_FILE_NAMES):
        if key not in state_dict:
            print(f"Warning: {key} not in model's state dict")
            continue
        
        state_tensor = state_dict[key]
        # Copy tensor data to the parameter/buffer (move to proper device if necessary)
        state_tensor.copy_(tensor.to(DEVICE))
        print('Model weight loaded:', key)
    print('All model weights loaded.')

    tokenized = tokenizer(SENTENCES)
    input_ids = torch.LongTensor(tokenized['input_ids']).to(DEVICE)
    attention_mask = torch.BoolTensor(tokenized['attention_mask']).to(DEVICE)

    print('Generating output token sequences...')
    output_token_sequences = generate(model, input_ids, attention_mask)
    print('Output token sequences generated.')

    for i, output_token_sequence in enumerate(output_token_sequences):
        print(f'Output token sequence {i}:', output_token_sequence)
        print(f'Decoded output token sequence {i}:', tokenizer.decode(output_token_sequence))