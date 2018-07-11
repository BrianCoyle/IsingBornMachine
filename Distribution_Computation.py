import numpy as np

import Amplitude_Computation_Hadamard_Test as amp


'''This function computes the output distribution after a given epoch to compute the KL Divergence'''
def compute_dist(N, N_v, N_h, J, b, gamma, bin_visible, bin_hidden):

	Output_Amplitude = np.zeros((2**N_v, 2**N_h), dtype = complex)

	#Define empty arrays, with each index, [string][h] corresponding to one possible string of
	#a string on the visibles, with one string of the hiddens, there are (2**N_v)*(2**N_h) possible binary strings
	Probs_Dist = np.zeros((2**N_v, 2**N_h))

	for v_string in range(0, 2**N_v):
		for h_string in range(0, 2**N_h):

			#Compute the amplitudes for each string of visible and hidden variables (with no gradient)
			Output_Amplitude[v_string, h_string] =  amp.Amplitude_Computation1(N, N_v, N_h, J, b, \
													gamma, bin_visible[v_string,:], bin_hidden[h_string, :])
			#print('\n The First Amplitude for epoch is ', bin_visible[v_string, :], bin_hidden[h_string, :],
			#' is: \n', Amp1[v_string, h_string], '\n')


	Probs_Dist = np.multiply(Output_Amplitude, Output_Amplitude.conjugate()).real

	#The distribution of the model over the visibles variables only
	P_v = Probs_Dist.sum(axis = 1)


	return  P_v
