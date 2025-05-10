from pathlib import Path
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
args = parser.parse_args()
local_dir = Path('.') / 'models' / args.model_id

# Load files
logits = torch.load(local_dir / 'logits.pt')
activations = torch.load(local_dir / 'activations.pt', weights_only=False)

# Print the shape of the logits and activations
print(logits.shape)
print(activations['blocks.11.ln1.hook_normalized'])

# Print an example of the activations