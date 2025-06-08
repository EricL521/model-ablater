from pathlib import Path
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model-id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
# --no-mapping or --mapping (default)
parser.add_argument('--mapping', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()
model_id = args.model_id
do_mapping = args.mapping
local_dir = Path('.') / 'models' / model_id

print("model_id:", model_id)
print("do_mapping:", do_mapping)

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
# 	'blocks.0.mlp.hook_pre_linear', 'blocks.0.hook_mlp_out', 'blocks.0.hook_mlp_out', 'blocks.0.hook_resid_post', 
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
from helper_functions.shown_layers import SHOWN_LAYERS
# Generate images in form of numpy array, one for each token in the input text
# I'm hardcoding values temporarily, but ideally it would be derived from the model
numpy_images = [np.zeros((0, 128, 3), dtype='uint8') for _ in range(logits.shape[1])]
SPACING_ALPHA = 200  # how bright spacing is between rows in the image
# Load mappings.npz
if do_mapping:
	MAPPINGS = np.load(local_dir / 'mappings.npz', allow_pickle=True)
	last_mapping = None
for key in activations.keys():
	layer_num_array = key.split('.')[:2:]
	layer_name = '.'.join(key.split('.')[2::])
	if not (layer_name in SHOWN_LAYERS or key in SHOWN_LAYERS):
		continue
	print(f"Processing layer: ({key})")
	numpy_activations = activations[key].detach().cpu().numpy()[0]
	if do_mapping:
		mapping_id = None
		shown_layer_key = key if key in SHOWN_LAYERS else layer_name
		if isinstance(SHOWN_LAYERS[shown_layer_key], str):
			mapping_id = SHOWN_LAYERS[shown_layer_key]\
				.replace('current', '.'.join(layer_num_array))\
				.replace('prev', '.'.join([layer_num_array[0], str(int(layer_num_array[1]) - 1)]))
		last_mapping = MAPPINGS.get(mapping_id, None)
		# Apply mapping if at least one dimension from numpy_activations matches the mapping
		if last_mapping is not None:
			print(f"Applying mapping of shape {last_mapping.shape} to activation of shape {numpy_activations.shape}")
			for i in range(numpy_activations.shape[0]):
				numpy_activations[i] = apply_mapping(numpy_activations[i], last_mapping)

	# Normalize the activations to [-255, 255] for image representation, with 0 mapped to 0
	norm_activations = np.zeros_like(numpy_activations)
	for i in range(numpy_activations.shape[0]):
		norm_activations[i] = numpy_activations[i] / np.max(np.abs(numpy_activations[i])) * 255
	# Convert to image
	if norm_activations.ndim == 2:
		# Make negative values red and positive values green (NOTE: green is 255*value, red is just value)
		colored_reshaped_norm_activations = np.zeros((norm_activations.shape[0], norm_activations.shape[1], 3), dtype='uint8')
		# Red channel is 255 for all negative values, green channel is 255 for all positive values, and the other two scale with magnitude
		colored_reshaped_norm_activations[:, :, 0] = np.where(norm_activations < 0, 255, 255 - norm_activations)
		colored_reshaped_norm_activations[:, :, 1] = np.where(norm_activations > 0, 255, 255 + norm_activations)
		colored_reshaped_norm_activations[:, :, 2] = 255 - np.abs(norm_activations)
		# resize to be 128 pixels wide
		colored_reshaped_norm_activations = colored_reshaped_norm_activations.reshape(colored_reshaped_norm_activations.shape[0], colored_reshaped_norm_activations.shape[1] // 128, 128, 3)
		# add to numpy_images
		for i in range(norm_activations.shape[0]):
			numpy_images[i] = np.vstack((numpy_images[i], colored_reshaped_norm_activations[i]))	
	elif norm_activations.ndim == 3:
		# if 4D, we are in a multi-head attention context
		# Make negative values red and positive values green (NOTE: green is 255*value, red is just value)
		colored_reshaped_norm_activations = np.zeros((norm_activations.shape[0], norm_activations.shape[1], norm_activations.shape[2], 3), dtype='uint8')
		# Red channel is 255 for all negative values, green channel is 255 for all positive values, and the other two scale with magnitude
		colored_reshaped_norm_activations[:, :, :, 0] = np.where(norm_activations < 0, 255, 255 - norm_activations)
		colored_reshaped_norm_activations[:, :, :, 1] = np.where(norm_activations > 0, 255, 255 + norm_activations)
		colored_reshaped_norm_activations[:, :, :, 2] = 255 - np.abs(norm_activations)
		# add to numpy_images
		if norm_activations.shape[0] == len(numpy_images):
			for i in range(norm_activations.shape[0]):
				numpy_images[i] = np.vstack((numpy_images[i], colored_reshaped_norm_activations[i]))
		# we are in a attention pattern context
		else:
			# add padding to center (24, 6, 6, 3) -> (128, 6, 6, 3)
			pad_total = 128 - colored_reshaped_norm_activations.shape[0]
			pad_left = pad_total // 2
			pad_right = pad_total - pad_left
			colored_reshaped_norm_activations = np.pad(
				colored_reshaped_norm_activations,
				((pad_left, pad_right), (0, 0), (0, 0), (0, 0)),
				mode='constant', constant_values=SPACING_ALPHA
			)
			# reorder axes
			colored_reshaped_norm_activations = colored_reshaped_norm_activations.transpose(1, 2, 0, 3)
			for i in range(norm_activations.shape[1]):
				numpy_images[i] = np.vstack((numpy_images[i], colored_reshaped_norm_activations[i]))
	# Add spacing between rows
	for i in range(len(numpy_images)):
		spacing = np.full((1, numpy_images[i].shape[1], 3), SPACING_ALPHA, dtype='uint8') 
		numpy_images[i] = np.vstack((numpy_images[i], spacing))
# Remove the last spacing row
numpy_images = [image[:-2] for image in numpy_images]

# Save images
os.makedirs(local_dir / 'activations', exist_ok=True)
for i, numpy_image in enumerate(numpy_images):
	image = Image.fromarray(numpy_image, mode='RGB')
	# rotate the image 90 degrees counterclockwise
	image = image.rotate(90, expand=True)
	# flip image vertically
	image = image.transpose(Image.FLIP_TOP_BOTTOM)
	# Save the image
	image.save(local_dir / 'activations' / f'token_{i}.png')
