from pathlib import Path
import torch
import torchlens as tl
from transformers import T5Tokenizer, T5ForConditionalGeneration
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--model_id', type=str, default='t5-small')
parser.add_argument('--text', type=str, default='translate English to German: How are you?')

args = parser.parse_args()

local_dir = Path('.') / 'models' / args.model_id
text = args.text

tokenizer = T5Tokenizer.from_pretrained(local_dir)
model = T5ForConditionalGeneration.from_pretrained(local_dir)
inputs = tokenizer(text, return_tensors="pt")

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]])

model_history = tl.log_forward_pass(model, inputs.input_ids,
									layers_to_save='all',
									vis_opt='none',
		 							input_kwargs={'decoder_input_ids': decoder_input_ids})
# Print all layer names
print(model_history.layer_labels)

'''
tensor([[-0.0690, -1.3957, -0.3231, -0.1980,  0.7197],
		[-0.1083, -1.5051, -0.2570, -0.2024,  0.8248],
		[ 0.1031, -1.4315, -0.5999, -0.4017,  0.7580],
		[-0.0396, -1.3813, -0.3523, -0.2008,  0.6654],
		[ 0.0980, -1.4073, -0.5934, -0.3866,  0.7371],
		[-0.1106, -1.2909, -0.3393, -0.2439,  0.7345]])
'''