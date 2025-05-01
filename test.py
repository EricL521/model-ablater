from pathlib import Path
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='t5-small')
args = parser.parse_args()
local_dir = Path('.') / 'models' / args.model_id

# Load model_history.npy
model_history = np.load(local_dir / 'model_history.npy', allow_pickle=True).item()
print(model_history['layer_labels'])
