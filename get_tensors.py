from pathlib import Path
import numpy as np
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
# Example usage
# print(model_history.layer_labels)
# print(model_history[output_1].tensor_contents)

# save the model history to a file using numpy
saved_object = {}
saved_object['layer_labels'] = model_history.layer_labels
for layer_label in model_history.layer_labels:
	saved_object[layer_label] = model_history[layer_label].tensor_contents
np.save(local_dir / 'model_history.npy', saved_object)
