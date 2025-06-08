import numpy as np

# returns a new np array with indices applied according to the mapping
def apply_mapping(np_array, mapping):
	new_array = np.zeros_like(np_array)
	if mapping.shape[0] == 1:
		# there is only one mapping, and it is not an attention layer
		for j in range(mapping.shape[1]):
			new_array[mapping[0, j]] = np_array[j]
		return new_array
	# there is more than one mapping, and it is an attention layer
	for i in range(np_array.shape[0]):
		for j in range(np_array.shape[1]):
			new_array[i, mapping[i // (np_array.shape[0] // mapping.shape[0]), j]] = np_array[i, j]
	return new_array
