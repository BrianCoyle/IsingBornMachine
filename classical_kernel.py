import numpy as np
from auxiliary_functions import ConvertToString, SampleArrayToList, L2NormForStrings


'''This function computes the classical mixture of Gaussian's kernel function between two sets of samples'''
def GaussianKernel(samples1, samples2, sigma, *argsv):
	#sigma[i] are bandwidth parameters
	# print(type(samples1) or type(samples2) is not np.ndarray)
	if type(samples1) is not np.ndarray and type(samples2) is not np.array:
		raise IOError('The input samples must be a numpy array')

	c = len(sigma)
	
	N_samples1 = samples1.shape[0]
	N_samples2 = samples2.shape[0]

	samples_list_1 = SampleArrayToList(samples1)
	samples_list_2 = SampleArrayToList(samples2)

	# gauss_kernel = np.zeros((N_samples1, N_samples2, c))
	# l2norm = np.zeros((N_samples1, N_samples2))

	gauss_kernel_sum_unnorm = 0
	# print(samples_list_1,'  ', samples_list_2)

	for sample1 in samples_list_1:
			for sample2 in  samples_list_2:
					# if (sample1 == sample2):

	# for i in range(0, N_samples1):
	# 	for j in range(0, N_samples2):

			# print(samples_list_1[sample1])

			# l2norm[i , j] = np.linalg.norm(samples1[i,:] - samples2[j, :], 2)
				gauss_sum_temp_unnorm = 0
				for k in range(0, c):
					gauss_sum_temp_unnorm = gauss_sum_temp_unnorm +\
					(1/c)*np.exp(-1/(2*sigma[k])*(L2NormForStrings(sample1, sample2)))
				# print(sample1, sasmple2, 'Kernel is:', gauss_sum_temp_unnorm)
				gauss_kernel_sum_unnorm = gauss_kernel_sum_unnorm + gauss_sum_temp_unnorm
	# print(gauss_kernel_sum_unnorm)
	if ('same' not in argsv):
		#if the two sets of samples come from the same distribution
		normed_gauss_kernel = (1/(N_samples1*(N_samples2)))*gauss_kernel_sum_unnorm
	else:
		normed_gauss_kernel = (1/(N_samples1*(N_samples2-1)))*gauss_kernel_sum_unnorm
	return normed_gauss_kernel

def GaussianKernelExact(N_v, bin_visible, sigma):
	#sigma[i] are bandwidth parameters
	c = len(sigma)
	N_strings  = 2**N_v

	gauss_kernel_exact_contribution = np.zeros((N_strings, N_strings, c))
	l2norm = np.zeros((N_strings, N_strings))
	gauss_kernel_exact_dict = {}
	for sample1 in range(0, N_strings):
		string1 = ConvertToString(sample1, N_v)
		for sample2 in range(0, sample1+1):
			string2 = ConvertToString(sample2, N_v)

			l2norm[sample1 , sample2] = np.linalg.norm(bin_visible[sample1,:] - bin_visible[sample2, :], 2)
			l2norm[sample2, sample1] = l2norm[sample1,sample2]
			
			for k in range(0, c):
				gauss_kernel_exact_contribution[sample1, sample2,k] = (1/c)*np.exp(-1/(2*sigma[k])*(l2norm[sample1, sample2]**2))

			gauss_kernel_exact = gauss_kernel_exact_contribution.sum(2)
			gauss_kernel_exact_dict[(string1, string2)] = gauss_kernel_exact[sample1, sample2]
			gauss_kernel_exact_dict[(string2, string1)] = gauss_kernel_exact_dict[(string1, string2)]

	return gauss_kernel_exact, gauss_kernel_exact_dict
