import numpy as np
from auxiliary_functions import SampleArrayToList, L2Norm

def GaussianKernel(sample1, sample2, sigma):
	'''This function computes a single Gaussian kernel value between two samples'''
	c = len(sigma)
	kernel = np.zeros((c))
	for k in range(0, c):
		kernel[k] = (1/c)*np.exp(-1/(2*sigma[k])*(L2Norm(sample1, sample2)))

	return kernel.sum()

def GaussianKernelArray(samples1, samples2, sigma):
	'''This function computes a kernel matrix for all pairs of samples, for a chosen kernel'''
	#sigma[i] are bandwidth parameters
	# if (type(samples1) is not np.ndarray or not list) or (type(samples2) is not np.array or not list):
	# 	raise IOError('The input samples must be a numpy array or list')
	if type(samples1) is np.ndarray:
		if samples1.ndim == 1: #Check if there is only a single sample in the array of samples
			N_samples1 = 1
		else:
			N_samples1 = samples1.shape[0]
	else: N_samples1 = len(samples1)

	if type(samples2) is np.ndarray:
		if samples2.ndim == 1:
			N_samples2 = 1
		else:
			N_samples2 = samples2.shape[0]
	else: N_samples2 = len(samples2)
	gauss_kernel_array = np.zeros((N_samples1, N_samples2))

	for sample1_index in range(0, N_samples1):
		for sample2_index in range(0, N_samples2):
			if N_samples1 == 1:
				gauss_kernel_array[sample1_index][sample2_index] =\
					GaussianKernel(samples1, samples2[sample2_index], sigma)
			elif N_samples2 == 1:
				gauss_kernel_array[sample1_index][sample2_index] =\
					GaussianKernel(samples1[sample1_index], samples2, sigma)
			else:
				gauss_kernel_array[sample1_index][sample2_index] =\
					GaussianKernel(samples1[sample1_index], samples2[sample2_index], sigma)

	return gauss_kernel_array


def GaussianKernelDict(samples1, samples2, sigma):
	'''This function computes a kernel matrix for all pairs of samples, for a chosen kernel'''
	#sigma[i] are bandwidth parameters

	if type(samples1) is not np.ndarray and type(samples2) is not np.array:
		raise IOError('The input samples must be a numpy array')

	if samples1.ndim == 1: #Check if there is only a single sample in the array of samples
		N_samples1 = 1
	else:
		N_samples1 = samples1.shape[0]

	if samples2.ndim == 1:
		N_samples2 = 1
	else:
		N_samples2 = samples2.shape[0]

	samples_list_1 = SampleArrayToList(samples1)
	samples_list_2 = SampleArrayToList(samples2)
	gauss_kernel_dict = {}

	for sample1_index in range(0, N_samples1):
			for sample2_index in range(0, N_samples2):
				
				gauss_kernel_dict[(samples_list_1[sample1_index], samples_list_2[sample2_index])] =\
					GaussianKernel(samples_list_1[sample1_index], samples_list_2[sample2_index], sigma)
	
	return gauss_kernel_dict