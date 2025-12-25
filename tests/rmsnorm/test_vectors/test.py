import torch
data = torch.load("model_analysis/model.layers.23.post_attention_layernorm.pt", map_location="cpu")
print(type(data))
print(data.keys() if isinstance(data, dict) else None)
