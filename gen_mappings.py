from pathlib import Path
import torch
import transformer_lens
import transformers
import argparse

# For detailed logs (INFO level)
transformers.logging.set_verbosity_info()

# For the most detailed logs (DEBUG level)
transformers.logging.set_verbosity_debug()

parser = argparse.ArgumentParser()

parser.add_argument('--model-id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

model_id = args.model_id
local_dir = Path('.') / 'models' / args.model_id
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

print("model_id:", model_id)
print("device:", device)

# Load the model into hooked transformer
tokenizer = transformers.AutoTokenizer.from_pretrained(local_dir)
hf_model = transformers.AutoModelForCausalLM.from_pretrained(local_dir, device_map=device, low_cpu_mem_usage=True)

# print(tokenizer.convert_ids_to_tokens(tokenizer("elephant")['input_ids']))
# ['<|begin_of_text|>', 'ele', 'phant']

model = transformer_lens.HookedTransformer.from_pretrained(
  model_id,
  hf_model=hf_model,
  device=device,
  tokenizer=tokenizer
)
del hf_model
model = model.to(device if torch.cuda.is_available() else "cpu")

# print(model.state_dict().keys())
# odict_keys(['embed.W_E', 
#   'blocks.0.attn.W_Q', 'blocks.0.attn.W_O', 'blocks.0.attn.b_Q', 'blocks.0.attn.b_O', 'blocks.0.attn._W_K', 
#   'blocks.0.attn._W_V', 'blocks.0.attn._b_K', 'blocks.0.attn._b_V', 'blocks.0.attn.mask', 'blocks.0.attn.IGNORE', 
#   'blocks.0.attn.rotary_sin', 'blocks.0.attn.rotary_cos', 'blocks.0.mlp.W_in', 'blocks.0.mlp.W_out', 'blocks.0.mlp.W_gate', 
#   'blocks.0.mlp.b_in', 'blocks.0.mlp.b_out', ... a lot of layers, ...
#   'blocks.27.mlp.b_out', 'unembed.W_U', 'unembed.b_U'])

# Neurons that are more strongly related (larger weight^2) should be closer together in generated images
import numpy as np
SHOWN_LAYER_WEIGHTS = {
	"attn.W_Q": ("attn.hook_q", "prev.mlp.W_out"), 
	"attn._W_K": ("attn.hook_k", "prev.mlp.W_out"),
	"attn._W_V": ("attn.hook_v", "prev.mlp.W_out"),
	"attn.W_O": ("hook_attn_out", "current.attn.hook_v"),
	"mlp.W_in": ("mlp.hook_pre_linear", "current.hook_attn_out"), 
	"mlp.W_out": ("mlp.hook_post", "current.mlp.hook_pre_linear"),
}
# Format of a mapping is [[old index -> new index]]
# Format of mappings is {layer name -> mapping}
mappings = {}
# returns a new tensor with indices applied according to the mapping
def apply_mapping(np_array, mapping):
	new_array = np.zeros_like(np_array)
	for i in range(mapping.shape[0]):
		for j in range(mapping.shape[1]):
			new_array[i, mapping[i, j]] = np_array[i, j]
	return new_array
def gen_mappings(in_weights=None, out_weights=None, prev_layer_mapping=None):
	if (in_weights is None and out_weights is None) or (in_weights is not None and out_weights is not None):
		raise ValueError("Exactly one of in_weights or out_weights should be provided.")
	used_weights = in_weights if in_weights is not None else out_weights
	used_weights = apply_mapping(used_weights, prev_layer_mapping) if prev_layer_mapping is not None else used_weights
	# calculate squared weights
	squared_weights = used_weights ** 2
	# calculate weighted average of indices based on squared weights
	weighted_indices = (squared_weights * np.arange(used_weights.shape[-2])[:, None]).sum(axis=-2) / squared_weights.sum(axis=-2)
	# sort indices to create a mapping
	sorted_indices = np.argsort(weighted_indices, axis=-1)
	if sorted_indices.ndim == 1:
		sorted_indices = sorted_indices[None, :]
	return sorted_indices

# Generate mappings for each layer in SHOWN_LAYER_WEIGHTS
for key in model.state_dict().keys():
	layer_num_array = key.split('.')[:2:]
	layer_name = '.'.join(key.split('.')[2::])
	if not layer_name in SHOWN_LAYER_WEIGHTS:
		continue
	print("Processing layer:", key)
	prev_layer_mapping_id = SHOWN_LAYER_WEIGHTS[layer_name][1]\
		.replace('current', '.'.join(layer_num_array))\
		.replace('prev', '.'.join([layer_num_array[0], str(int(layer_num_array[1]) - 1)]))
	mappings['.'.join('.'.join(layer_num_array) + SHOWN_LAYER_WEIGHTS[layer_name][0])] = gen_mappings(
		in_weights=model.state_dict()[key].detach().cpu().numpy(),
		prev_layer_mapping=mappings.get(prev_layer_mapping_id, None)
	)

# save mappings to a file using numpy
mappings_file = local_dir / 'mappings'
np.savez(mappings_file, mappings)