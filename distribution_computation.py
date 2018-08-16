import numpy as np

import amplitude_computation_hadamard_test as amp


'''This function computes the output distribution after a given epoch to compute the KL Divergence'''
def ComputeDist(N, N_v, N_h, J, b, gamma, bin_visible, bin_hidden):

	output_amplitude = np.zeros((2**N_v, 2**N_h), dtype = complex)

	#Define empty arrays, with each index, [string][h] corresponding to one possible string of
	#a string on the visibles, with one string of the hiddens, there are (2**N_v)*(2**N_h) possible binary strings
	probs_dist = np.zeros((2**N_v, 2**N_h))

	for v_string in range(0, 2**N_v):
		for h_string in range(0, 2**N_h):

			#Compute the amplitudes for each string of visible and hidden variables (with no gradient)
			output_amplitude[v_string, h_string] =  amp.AmplitudeComputation1(N, N_v, N_h, J, b, \
													gamma, bin_visible[v_string,:], bin_hidden[h_string, :])

	probs_dist = np.multiply(output_amplitude, output_amplitude.conjugate()).real

	#The distribution of the model over the visibles variables only
	p_v = probs_dist.sum(axis = 1)
	
	return  p_v
