from pathlib import Path
import torch
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
args = parser.parse_args()
model_id = args.model_id
local_dir = Path('.') / 'models' / model_id

print("model_id:", model_id)

# Load files
logits = torch.load(local_dir / 'logits.pt')
activations = torch.load(local_dir / 'activations.pt', weights_only=False)

# print(logits.shape)
# torch.Size([1, 3, 128256])

# print(activations)
# ActivationCache with keys ['hook_embed', 
# 	'blocks.0.hook_resid_pre', 'blocks.0.ln1.hook_scale', 'blocks.0.ln1.hook_normalized', 'blocks.0.attn.hook_q', 
# 	'blocks.0.attn.hook_k', 'blocks.0.attn.hook_v', 'blocks.0.attn.hook_rot_q', 'blocks.0.attn.hook_rot_k', 
# 	'blocks.0.attn.hook_attn_scores', 'blocks.0.attn.hook_pattern', 'blocks.0.attn.hook_z', 'blocks.0.hook_attn_out', 
# 	'blocks.0.hook_resid_mid', 'blocks.0.ln2.hook_scale', 'blocks.0.ln2.hook_normalized', 'blocks.0.mlp.hook_pre', 
# 	'blocks.0.mlp.hook_pre_linear', 'blocks.0.mlp.hook_post', 'blocks.0.hook_mlp_out', 'blocks.0.hook_resid_post', 
# 	... a lot of layers, ... blocks.27.hook_resid_post', 'ln_final.hook_scale', 'ln_final.hook_normalized']

# NOTE: in the following logs, n is the number of tokens in the input text

# print(activations['blocks.0.ln1.hook_normalized'].shape)
# torch.Size([1, n, 3072])
# print(activations['blocks.0.attn.hook_q'].shape)
# torch.Size([1, n, 24, 128])
# print(activations['blocks.0.attn.hook_k'].shape)
# torch.Size([1, n, 8, 128])
# print(activations['blocks.0.attn.hook_v'].shape)
# torch.Size([1, n, 8, 128])
# print(activations['blocks.0.attn.hook_rot_q'].shape)
# torch.Size([1, n, 24, 128])
# print(activations['blocks.0.attn.hook_attn_scores'].shape)
# torch.Size([1, 24, n, n])
# print(activations['blocks.0.attn.hook_pattern'].shape)
# torch.Size([1, 24, n, n])
# print(activations['blocks.0.attn.hook_z'].shape)
# torch.Size([1, n, 24, 128]))
# print(activations['blocks.0.hook_attn_out'].shape)
# torch.Size([1, n, 3072])

# print(activations['blocks.0.hook_mlp_out'].shape)
# torch.Size([1, n, 3072])

# print(activations['ln_final.hook_scale'].shape)
# torch.Size([1, n, 1])

# print(activations['ln_final.hook_normalized'])
# torch.Size([1, n, 3072])

# How to get output logits from activations
import transformers
tokenizer = transformers.AutoTokenizer.from_pretrained(local_dir)
# logits = model.lm_head(activations['ln_final.hook_normalized'])
token_ids = logits.argmax(dim=-1)
output_tokens = tokenizer.convert_ids_to_tokens(token_ids[0])
print(output_tokens)
# ['The', 'Ġoverview', "'s", 'Ġa', 'Ġlarge', '3']

# Use activations to generate an image
image = Image.new('RGB', (512, 512))
