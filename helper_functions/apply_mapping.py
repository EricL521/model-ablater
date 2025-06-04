import numpy as np

# returns a new np array with indices applied according to the mapping
def apply_mapping(np_array, mapping):
	new_array = np.zeros_like(np_array)
	if np_array.ndim == 2:
		# the input is 2D, and it is not an attention layer
		if mapping.shape[0] != 1:
			raise ValueError("Mapping for 2D array should have shape (1, n), but got shape {}".format(mapping.shape))
		for j in range(mapping.shape[1]):
			new_array[mapping[0, j]] = np_array[j]
		return new_array
	# if the input is 3D, then it is an attention layer
	for i in range(mapping.shape[0]):
		for j in range(mapping.shape[1]):
			new_array[i, mapping[i, j]] = np_array[i, j]
	return new_array
