# Returns the position in the original array that corresponds to the position in the new array
def unmap_position(mapped_position, mapping, np_array_shape):
	if mapping.shape[0] == 1:
		# there is only one mapping, and it is not an attention layer
		for j in range(mapping.shape[1]):
			if mapping[0, j] == mapped_position[1]:
				return (mapped_position[0], j)
		return None
	# there is more than one mapping, and it is an attention layer
	for j in range(mapping.shape[1]):
		if mapping[mapped_position[1] // (np_array_shape[1] // mapping.shape[0]), j] == mapped_position[2]:
			return (mapped_position[0], mapped_position[1], j)
	return None
