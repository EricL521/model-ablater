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

parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
parser.add_argument('--text', type=str, default='An elephant is a')
parser.add_argument('--device', type=str, default='cpu')

args = parser.parse_args()

model_id = args.model_id
local_dir = Path('.') / 'models' / args.model_id
text = args.text
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

# Load the model into hooked transformer
tokenizer = transformers.AutoTokenizer.from_pretrained(local_dir)
hf_model = transformers.AutoModelForCausalLM.from_pretrained(local_dir, device_map=device, low_cpu_mem_usage=True)

model = transformer_lens.HookedTransformer.from_pretrained(
  model_id,
  hf_model=hf_model,
  device=device,
  tokenizer=tokenizer
)
model = model.to(device if torch.cuda.is_available() else "cpu")

# Run model with transformer_lens
logits, activations = model.run_with_cache(text)

print(logits, activations)

# save the model history to files
torch.save(logits, local_dir / 'logits.pt')
torch.save(activations, local_dir / 'activations.pt')
