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
# --no-calc or --calc (default)
parser.add_argument('--calc', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--text', type=str, default='an elephant is a ')
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

do_calc = args.calc
model_id = args.model_id
local_dir = Path('.') / 'models' / args.model_id
text = args.text
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

print("model_id:", model_id)
print("do_calc:", do_calc)
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

# print(model.state_dict().keys())
# odict_keys(['embed.W_E', 
#   'blocks.0.attn.W_Q', 'blocks.0.attn.W_O', 'blocks.0.attn.b_Q', 'blocks.0.attn.b_O', 'blocks.0.attn._W_K', 
#   'blocks.0.attn._W_V', 'blocks.0.attn._b_K', 'blocks.0.attn._b_V', 'blocks.0.attn.mask', 'blocks.0.attn.IGNORE', 
#   'blocks.0.attn.rotary_sin', 'blocks.0.attn.rotary_cos', 'blocks.0.mlp.W_in', 'blocks.0.mlp.W_out', 'blocks.0.mlp.W_gate', 
#   'blocks.0.mlp.b_in', 'blocks.0.mlp.b_out', ... a lot of layers, ...
#   'blocks.27.mlp.b_out', 'unembed.W_U', 'unembed.b_U'])

# state * weight_matrix

# print(model.state_dict()['blocks.0.attn.W_Q'].shape)
# torch.Size([24, 3072, 128])

# print(model.state_dict()['blocks.0.attn.W_O'].shape)
# torch.Size([24, 128, 3072])

# print(model.state_dict()['blocks.0.mlp.W_in'].shape)
# torch.Size([3072, 8192])

# print(model.state_dict()['blocks.0.mlp.W_gate'].shape)
# torch.Size([3072, 8192])

# print(model.state_dict()['blocks.26.mlp.W_out'].shape)
# torch.Size([8192, 3072])

# print(model.state_dict()['unembed.W_U'].shape)
# torch.Size([3072, 128256])

# Run model with transformer_lens
if do_calc:
  logits, activations = model.run_with_cache(text)

  print(logits, activations)

  # save the model history to files
  torch.save(logits, local_dir / 'logits.pt')
  torch.save(activations, local_dir / 'activations.pt')
