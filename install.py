from huggingface_hub import snapshot_download
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.2-3B-Instruct')

args = parser.parse_args()

model_id = args.model_id
local_dir = Path('.') / 'models' / args.model_id

# Download the model to the specified directory
snapshot_download(repo_id=model_id, local_dir=local_dir, local_dir_use_symlinks=False)
