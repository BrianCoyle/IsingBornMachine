import numpy as np
from numpy import linalg as LA


'''This function computes the classical mixture of Gaussian's kernel function between two sets of samples'''

def gaussian_kernel(sample1, sample2, sigma):
	#sigma[i] are bandwidth parameters
	
	c = len(sigma)
	N_samples1  = sample1.shape[0]
	N_samples2  = sample2.shape[0]

	gauss_kernel = np.zeros((N_samples1, N_samples2, c))
	l2norm = np.zeros((N_samples1, N_samples2))

	for i in range(0, N_samples1):
		for j in range(0, N_samples2):

			l2norm[i , j] = LA.norm(sample1[i,:] - sample2[j, :], 2)

			for k in range(0, c):

				gauss_kernel[i,j,k] = (1/c)*np.exp(-1/(2*sigma[k])*(l2norm[i,j]**2))
	gausssum = gauss_kernel.sum(2)
	#print(gausssum)
	return gauss_kernel.sum(2)



