
import numpy as np

from pyquil.api import get_qc
from classical_kernel import GaussianKernelArray, GaussianKernel
from quantum_kernel import  QuantumKernelArray, QuantumKernel

from kernel_functions import NormaliseKernel
from file_operations_in import KernelDictFromFile, DataDictFromFile
from auxiliary_functions import ShiftString, EmpiricalDist, FindNumQubits, ToString
 
import stein_score as ss
from spectral_stein_score import SpectralSteinScore
import json	

def DeltaTerms(device_params, kernel_choice, kernel_array, N_samples, sample_array_1, sample_array_2, flag):
	'''This kernel computes the shifted *Delta* terms used in the Stein Discrepancy'''
	N_qubits = FindNumQubits(device_params)
	N_samples1 = len(sample_array_1)
	N_samples2 = len(sample_array_2)

	kernel_x_shifted, kernel_y_shifted, kernel_xy_shifted = [np.zeros((N_samples1, N_samples2, N_qubits)) for _ in range(3)]
	delta_x_kernel, delta_y_kernel, delta_xy_kernel = [np.zeros((N_samples1, N_samples2, N_qubits)) for _ in range(3)]

	for sample_index1 	in range(0, N_samples1):
		for sample_index2 in range(0, N_samples2):
			for qubit in range(0, N_qubits):
            
				if flag == 'Onfly': 
					kernel =  kernel_array[sample_index1][sample_index2]
					sample1 = sample_array_1[sample_index1]
					sample2 = sample_array_2[sample_index2]
					shiftedsample1 = ShiftString(sample1, qubit)
					shiftedsample2 = ShiftString(sample2, qubit)

					if (kernel_choice == 'Gaussian'):

						sigma = np.array([0.25, 10, 1000])

						kernel_x_shifted[sample_index1, sample_index2, qubit]   = GaussianKernel(shiftedsample1,    sample2,        sigma)
						kernel_y_shifted[sample_index1, sample_index2, qubit]   = GaussianKernel(sample1,           shiftedsample2, sigma)
						kernel_xy_shifted[sample_index1, sample_index2, qubit]  = GaussianKernel(shiftedsample1,    shiftedsample2, sigma)

				elif flag == 'Precompute': #Use Precomputed kernel dictionary to accelerate training
					kernel_dict  = KernelDictFromFile(device_params, N_samples, kernel_choice)
					kernel = kernel_dict[(sample1, sample2)]

					sample1 		= ToString(sample_array_1[sample_index1])
					sample2 		= ToString(sample_array_2[sample_index2])
					shiftedsample1 	= ToString(ShiftString(sample1, qubit))
					shiftedsample2 	= ToString(ShiftString(sample2, qubit))

					kernel_x_shifted[sample_index1][sample_index2][qubit]    = kernel_dict[(shiftedsample1, sample2)]
					kernel_y_shifted[sample_index1][sample_index2][qubit]    = kernel_dict[(sample1,        shiftedsample2)]
					kernel_xy_shifted[sample_index1][sample_index2][qubit]   = kernel_dict[(shiftedsample1, shiftedsample2)]

					delta_x_kernel[sample_index1][sample_index2][qubit] =  kernel - kernel_x_shifted[sample_index1, sample_index2, qubit]                                           
					delta_y_kernel[sample_index1][sample_index2][qubit] =  kernel - kernel_y_shifted[sample_index1, sample_index2, qubit]          
					delta_xy_kernel[sample_index1][sample_index2][qubit] = kernel - kernel_xy_shifted[sample_index1, sample_index2, qubit]                                           
									
				trace = N_qubits*kernel_array - kernel_x_shifted.sum(axis = 2) - kernel_y_shifted.sum(axis = 2) + kernel_xy_shifted.sum(axis = 2)

	return delta_x_kernel, delta_y_kernel, trace

def WeightedKernel(device_params, kernel_choice, kernel_array, N_samples, data_samples, data_probs, sample_array_1, sample_array_2, stein_params, flag, *argsv):
	'''This kernel computes the weighted kernel for all samples from the two distributions sample_list_1, sample_list_2'''

	delta_x_kernel, delta_y_kernel, trace = DeltaTerms(device_params, kernel_choice, kernel_array, N_samples, sample_array_1, sample_array_2, flag)

	#Parameters required for computing the Stein Score
	score_approx        = stein_params[0]
	J                   = stein_params[1]
	chi                 = stein_params[2]
	stein_kernel_choice = stein_params[3]
	stein_sigma         = stein_params[4]


	if (score_approx == 'Exact_Score'):
		stein_score_matrix_1 = ss.MassSteinScore(sample_array_1, data_probs)
		stein_score_matrix_2 = ss.MassSteinScore(sample_array_2, data_probs)
	elif (score_approx == 'Identity_Score'):
		stein_score_matrix_1 = ss.IdentitySteinScore(data_samples, stein_kernel_choice, chi, stein_sigma)
		stein_score_matrix_2 = ss.IdentitySteinScore(data_samples, stein_kernel_choice, chi, stein_sigma)
	elif (score_approx == 'Spectral_Score'):
		#compute score matrix using spectral method for all samples, x and y according to the data distribution.
		stein_score_matrix_1 = SpectralSteinScore(sample_array_1, data_samples, J, stein_sigma)
		stein_score_matrix_2 = SpectralSteinScore(sample_array_2, data_samples, J, stein_sigma)

	else: raise IOError('Please enter \'Exact_Score\', \'Identity_Score\' or \'Spectral_Score\' for score_approx')

	# print('Score Matrix 1 is:', stein_score_matrix_1)
	N_samples1 = len(sample_array_1)
	N_samples2 = len(sample_array_2)

	weighted_kernel = np.zeros((N_samples1, N_samples2))

	for sample_index1 in range(0, N_samples1):
		for sample_index2 in range(0, N_samples2):

			delta_x =   np.transpose(delta_x_kernel[sample_index1][sample_index2])
			delta_y =   delta_y_kernel[sample_index1][sample_index2]
			kernel  =   kernel_array[sample_index1][sample_index2]

			if sample_index1 == sample_index2:
				if 'same' in argsv: #if 'same' in optional args, do not contributions that are computed with themselves
					weighted_kernel[sample_index1, sample_index2] = 0
			else: 
				weighted_kernel[sample_index1, sample_index2] =                                                                 \
				np.dot(np.transpose(stein_score_matrix_1[sample_index1]), np.dot(kernel, stein_score_matrix_2[sample_index2]))  \
				- np.dot(np.transpose(stein_score_matrix_1[sample_index1]), delta_y)                                            \
				- np.dot(delta_x, stein_score_matrix_2[sample_index2])                                                          \
				+ trace[sample_index1][sample_index2]

	return weighted_kernel
