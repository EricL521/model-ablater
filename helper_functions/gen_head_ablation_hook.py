def gen_head_ablation_hook(head_index_to_ablate):
	def head_ablation_hook(value, hook):
		# print(f"Shape of the value tensor: {value.shape}, head_index_to_ablate: {head_index_to_ablate}")
		if isinstance(head_index_to_ablate, int) or len(head_index_to_ablate) == 1:
			value[:, :, head_index_to_ablate] = 0.
		else:
			value[:, :, head_index_to_ablate[0], head_index_to_ablate[1]] = 0.
		return value
	return head_ablation_hook
