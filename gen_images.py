from pathlib import Path
import torch
import argparse

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
from PIL import Image
import os
import numpy as np
from helper_functions.apply_mapping import apply_mapping
# Generate images, one for each layer
images = {}
# Define which layers to show
SHOWN_LAYERS = {
	"ln1.hook_normalized": "prev.mlp.hook_post",
	"attn.hook_q": "current.attn.hook_q", 
	"attn.hook_k": "current.attn.hook_k", 
	"attn.hook_v": "current.attn.hook_v", 
	"attn.hook_rot_q": "current.attn.hook_q", 
	"attn.hook_rot_k": "current.attn.hook_k", 
	"attn.hook_pattern": None, 
	"attn.hook_z": "current.attn.hook_v", 
	"hook_attn_out": "prev.mlp.hook_post",
	"ln2.hook_normalized": "prev.mlp.hook_post", 
	"mlp.hook_pre_linear": "current.mlp.hook_pre_linear", 
	"mlp.hook_post": "current.mlp.hook_post", 
	"hook_mlp_out": "current.mlp.hook_post", 
	"hook_resid_post": "current.mlp.hook_post",
}
SPACING_ALPHA = 200  # how bright spacing is between rows in the image
# Load mappings.npz
MAPPINGS = np.load(local_dir / 'mappings.npz', allow_pickle=True)
index = 0
last_mapping = None
for key in activations.keys():
	layer_num_array = key.split('.')[:2:]
	layer_name = '.'.join(key.split('.')[2::])
	if not layer_name in SHOWN_LAYERS:
		continue
	print(f"Processing layer: ({key})")
	numpy_activations = activations[key].detach().cpu().numpy()[0]
	mapping_id = None
	if isinstance(SHOWN_LAYERS[layer_name], str):
		mapping_id = SHOWN_LAYERS[layer_name]\
			.replace('current', '.'.join(layer_num_array))\
			.replace('prev', '.'.join([layer_num_array[0], str(int(layer_num_array[1]) - 1)]))
	last_mapping = MAPPINGS.get(mapping_id, None)
	# Apply mapping if at least one dimension from numpy_activations matches the mapping
	last_mapping = None
	if last_mapping is not None:
		for i in range(numpy_activations.shape[0]):
			numpy_activations[i] = apply_mapping(numpy_activations[i], last_mapping)
	
	# Normalize the activations to [-255, 255] for image representation, with 0 mapped to 0
	norm_activations = numpy_activations / np.max(np.abs(numpy_activations)) * 255
	# Convert to image
	if norm_activations.ndim == 2:
		reshaped_norm_activations = norm_activations.reshape(norm_activations.shape[0], norm_activations.shape[1])
		# Make negative values red and positive values green (NOTE: green is 255*value, red is just value)
		colored_reshaped_norm_activations = np.zeros((reshaped_norm_activations.shape[0], reshaped_norm_activations.shape[1], 3), dtype='uint8')
		# Red channel is 255 for all negative values, green channel is 255 for all positive values, and the other two scale with magnitude
		colored_reshaped_norm_activations[:, :, 0] = np.where(reshaped_norm_activations < 0, 255, 255 - reshaped_norm_activations)
		colored_reshaped_norm_activations[:, :, 1] = np.where(reshaped_norm_activations > 0, 255, 255 + reshaped_norm_activations)
		colored_reshaped_norm_activations[:, :, 2] = 255 - np.abs(reshaped_norm_activations)
		# insert empty row between each row to make it more readable
		spaced_colored_reshaped_norm_activations = np.insert(colored_reshaped_norm_activations, np.arange(1, colored_reshaped_norm_activations.shape[0]), SPACING_ALPHA, axis=0)
		# convert to image
		image = Image.fromarray(spaced_colored_reshaped_norm_activations, mode='RGB')
	elif norm_activations.ndim == 3:
		# if 4D, we are in a multi-head attention context
		# We will combine the heads side by side, i.e. (6, 24, 128) -> (6 * 24, 128)
		reshaped_norm_activations = norm_activations.reshape(norm_activations.shape[0] * norm_activations.shape[1], norm_activations.shape[2])
		# Make negative values red and positive values green (NOTE: green is 255*value, red is just value)
		colored_reshaped_norm_activations = np.zeros((reshaped_norm_activations.shape[0], reshaped_norm_activations.shape[1], 3), dtype='uint8')
		# Red channel is 255 for all negative values, green channel is 255 for all positive values, and the other two scale with magnitude
		colored_reshaped_norm_activations[:, :, 0] = np.where(reshaped_norm_activations < 0, 255, 255 - reshaped_norm_activations)
		colored_reshaped_norm_activations[:, :, 1] = np.where(reshaped_norm_activations > 0, 255, 255 + reshaped_norm_activations)
		colored_reshaped_norm_activations[:, :, 2] = 255 - np.abs(reshaped_norm_activations)
		# insert empty row between each block to make it more readable
		spaced_colored_reshaped_norm_activations = np.insert(colored_reshaped_norm_activations, np.arange(1, reshaped_norm_activations.shape[0] // norm_activations.shape[1]) * norm_activations.shape[1], SPACING_ALPHA, axis=0)
		# convert to image
		image = Image.fromarray(spaced_colored_reshaped_norm_activations, mode='RGB')
	# save image
	images[str(index) + "_" + key] = image
	index += 1

# Save images
os.makedirs(local_dir / 'layers', exist_ok=True)
for key, image in images.items():
	image.save(local_dir / 'layers' / f"{key}.png")

