from pathlib import Path
import torch
import transformer_lens
import transformers
import argparse
import numpy as np

# For detailed logs (INFO level)
transformers.logging.set_verbosity_info()

# For the most detailed logs (DEBUG level)
transformers.logging.set_verbosity_debug()

parser = argparse.ArgumentParser()

parser.add_argument('--model-id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
parser.add_argument('--text', type=str, default='An elephant is a large')
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

model_id = args.model_id
local_dir = Path('.') / 'models' / args.model_id
text = args.text
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

print("model_id:", model_id)
print("text:", text)
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

def gen_head_ablation_hook(head_index_to_ablate):
	def head_ablation_hook(value, hook):
		# print(f"Shape of the value tensor: {value.shape}")
		if isinstance(head_index_to_ablate, int) or len(head_index_to_ablate) == 1:
			value[:, :, head_index_to_ablate] = 0.
		else:
			# print(head_index_to_ablate)
			value[:, head_index_to_ablate[0], head_index_to_ablate[1]] = 0.
		return value
	return head_ablation_hook

# Run model with transformer_lens
def run_model(model, text, ablate_indices=None):
	# We define a head ablation hook
	tokens = model.to_tokens(text)
	logits, loss = model.run_with_hooks(
		tokens, return_type='both',
		fwd_hooks=[(x[0], gen_head_ablation_hook(x[1])) for x in ablate_indices] if ablate_indices else []
	)
	token_ids = logits.argmax(dim=-1)
	output_text = model.to_str_tokens(token_ids)
	input_text = model.to_str_tokens(tokens)
	return input_text, output_text, loss

# Test with blocking:
# (blocks.12.hook_mlp_out, 3039)
# print(run_model(model, text, [("blocks.12.hook_mlp_out", 3039)]))

# Load selected activations
selected_activations = {eval(k): v for k, v in np.load(local_dir / 'selected_activations.npz', allow_pickle=True).items()}
print(run_model(model, text, selected_activations.keys()))
