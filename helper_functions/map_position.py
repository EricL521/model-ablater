# Returns the position in the mapped array that corresponds to the position in the original array
def map_position(mapped_position, mapping, np_array_shape):
	if mapping.shape[0] == 1:
		# there is only one mapping, and it is not an attention layer
		return (mapped_position[0], mapping[0, mapped_position[1]])
	# there is more than one mapping, and it is an attention layer
	return (mapped_position[0], mapped_position[1], mapping[mapped_position[1] // (np_array_shape[1] // mapping.shape[0]), mapped_position[2]])
