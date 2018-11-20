import numpy as np
from random import *

from classical_kernel import GaussianKernel, GaussianKernelExact
from file_operations_in import KernelDictFromFile, DataDictFromFile
from quantum_kernel import  QuantumKernelComputation, EncodingFunc

from auxiliary_functions import ConvertToString, EmpiricalDist, SampleArrayToList
import sys
import json


def SumContribution(N_qubits, dict_one, dict_two, kernel_dict, *argsv):
	'''
	Computes the contribution to the MMD/MMD gradient for two sets of probabilities, dict_one 
	and dict_two'''
	N_probs_one = len(dict_one)
	N_probs_two = len(dict_two)

	first_second = np.zeros((N_probs_one, N_probs_one))

	for first_term in range(0, N_probs_one):
		first_string = ConvertToString(first_term, N_qubits)
		for second_term in range(0, N_probs_two):
			second_string = ConvertToString(second_term, N_qubits)
			if (first_string == second_string):
				# if ('same' not in argsv):
					first_second[first_term, second_term] = dict_one[first_string]\
										*dict_two[second_string]*kernel_dict[(first_string, second_string)]
			else:
				first_second[first_term, second_term] = dict_one[first_string]\
										*dict_two[second_string]*kernel_dict[(first_string, second_string)]
	return first_second
	
def KernelSum(N_qubits, samplearray1, samplearray2, kernel_sampled_dict, *argsv):
	'''This function computes the contribution to the MMD from the empirical distibutions
			from two sets of samples'''

	N_samples1 = samplearray1.shape[0]
	N_samples2 = samplearray2.shape[0]

	empiricaldist_dict1 = EmpiricalDist(samplearray1, N_qubits)
	empiricaldist_dict2 = EmpiricalDist(samplearray2, N_qubits)
	
	samples_list_1 = SampleArrayToList(samplearray1)
	samples_list_2 = SampleArrayToList(samplearray2)
	# print(samples_list_1,'  ', samples_list_2)
	kernel_sum_unnorm = 0
	for sample1 in samples_list_1:
		for sample2 in  samples_list_2:
			# print(sample1,'  ', sample2)

			# if (sample1 == sample2):
				# if ('same' not in argsv):
			kernel_sum_unnorm = kernel_sum_unnorm \
			+ kernel_sampled_dict[sample1, sample2]
			# else:
				# kernel_sum_unnorm = kernel_sum_unnorm \
				# 	+ kernel_sampled_dict[sample1, sample2]
	# print(kernel_sum_unnorm)
	if ('same' not in argsv):
		kernel_sum = (1/(N_samples1*N_samples2))*kernel_sum_unnorm
	else: 
		kernel_sum = (1/(N_samples1*(N_samples2-1)))*kernel_sum_unnorm

	#Using the probabilities method
	# kernel_prob = SumContribution(N_qubits, empiricaldist_dict1 ,empiricaldist_dict2, kernel_sampled_dict)
	# print('kernel_sum is:\n', kernel_sum, '\nThe kernel prob summed is:\n', kernel_prob.sum())

	return kernel_sum

def MMDGrad(N_qubits, data, data_exact_dict,
						born_samples, born_probs_dict,
						born_samples_plus, born_plus_exact_dict, 
						born_samples_minus, born_minus_exact_dict,
						N_samples, kernel_choice, approx, flag):

	
	if (flag == 'Onfly'):
		if (approx == 'Sampler'):
			if (kernel_choice == 'Gaussian'):
				sigma = np.array([0.25, 10, 1000])

				#Compute the Gaussian kernel on the fly for all pairs of samples required
				gaussian_born_plus = GaussianKernel(born_samples, born_samples_plus, sigma)	
				gaussian_born_minus = GaussianKernel(born_samples, born_samples_minus, sigma)	
				gaussian_data_plus = GaussianKernel(data, born_samples_plus, sigma)		
				gaussian_data_minus = GaussianKernel(data, born_samples_minus, sigma)
				# print('This is the onfly gaussian_born_plus:', gaussian_data_minus)

				L_MMD_Grad = 2*(gaussian_born_minus - gaussian_born_plus-gaussian_data_minus + gaussian_data_plus)

		else: raise IOError('\'approx must be \'Sampler\' to run training with on-the-fly kernel computation\'')

	elif (flag == 'Precompute'):
		if (approx == 'Sampler'):
			
			kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_samples, kernel_choice)

			k_bornplus_born = KernelSum(N_qubits, born_samples, born_samples_plus,kernel_sampled_dict)
			k_bornminus_born = KernelSum(N_qubits, born_samples, born_samples_minus,kernel_sampled_dict)
			k_bornplus_data = KernelSum(N_qubits, data, born_samples_plus,kernel_sampled_dict)
			k_bornminus_data = KernelSum(N_qubits, data, born_samples_minus,kernel_sampled_dict)

		elif (approx == 'Exact'):

			#We have dictionaries of 5 files, the kernel, the data, the born probs, the born probs plus/minus
			kernel_dict = KernelDictFromFile(N_qubits, N_samples, kernel_choice)
			#compute the term kp_1p_2 to add to expectation value for each pair, from bornplus/bornminus, with born machine and data
			born_born_plus = SumContribution(N_qubits, born_probs_dict, born_plus_exact_dict, kernel_dict)
			born_born_minus = SumContribution(N_qubits, born_probs_dict, born_minus_exact_dict, kernel_dict)
			data_born_plus = SumContribution(N_qubits, data_exact_dict, born_plus_exact_dict, kernel_dict)
			data_born_minus = SumContribution(N_qubits, data_exact_dict, born_minus_exact_dict, kernel_dict)

			#Return the gradient of the loss function (L = MMD^2) for a given parameter
		# L_MMD_Grad1 = 2*(born_born_minus.sum()-born_born_plus.sum()-data_born_minus.sum()+data_born_plus.sum())
		L_MMD_Grad =  2*(k_bornminus_born- k_bornplus_born- k_bornminus_data + k_bornplus_data)

	# print('\nHEREE : L_MMD_Grad is:\n', L_MMD_Grad)
	return L_MMD_Grad



def MMDCost(N_qubits, data, data_exact_dict, born_samples, born_probs_dict,	
			N_samples, kernel_choice, approx, flag):

	'''This function computes the MMD cost function between P and Q from samples from P and
		Q if the sampling approx is taken, otherwise if the cost function is to be computed exactly'''

	if (flag == 'Onfly'):
		if (approx == 'Sampler'):
			if (kernel_choice == 'Gaussian'):

				sigma = np.array([0.25, 10, 1000])

				#Compute the Gaussian kernel on the fly for all samples in the sample space
				gaussian_born_born = GaussianKernel(born_samples, born_samples, sigma, 'same')	

				gaussian_born_data = GaussianKernel(born_samples, data, sigma)	
	
				gaussian_data_data = GaussianKernel(data, data, sigma, 'same')		
				# print(gauss)
				L_mmd =  gaussian_born_born - 2*gaussian_born_data + gaussian_data_data
		else: raise IOError('\'approx must be \'Sampler\' to run training with on-the-fly kernel computation\'')

	elif (flag == 'Precompute'):
		if (approx == 'Sampler'):
			#Compute the empirical data distibution given samples
		
			kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_samples, kernel_choice)

			k_bb= KernelSum(N_qubits, born_samples, born_samples, kernel_sampled_dict,'same')
			# print('HELLO',k_bb, born_born.sum())
			k_bd = KernelSum(N_qubits, born_samples, data, kernel_sampled_dict)
			k_dd = KernelSum(N_qubits, data, data, kernel_sampled_dict, 'same')


			#L1 = k_bb + k_dd - 2*(k_bd)
			L_mmd =  k_bb - 2*k_bd + k_dd
			# print('This is the precomputed mmd:', L_mmd)

		elif (approx == 'Exact'):

			#compute the term kp_1p_2 to add to expectation value for each pair, from born/bornplus/bornminus machine, and data
			k_exact_dict = KernelDictFromFile(N_qubits, 'infinite', kernel_choice)

			born_born = SumContribution(N_qubits, born_probs_dict, born_probs_dict, k_exact_dict,'same')
			born_data = SumContribution(N_qubits, data_exact_dict, born_probs_dict, k_exact_dict)
			data_data = SumContribution(N_qubits, data_exact_dict, data_exact_dict, k_exact_dict, 'same')
			# print(born_born)
			# print('\n', data_data, '\n', born_data)
			#return the loss function (L = MMD^2)
			L_mmd = born_born.sum() - 2*born_data.sum() + data_data.sum()

	else: raise IOError('\'flag\' must be either \'Onfly\' or \'Precompute\'')
	return L_mmd


def MMDKernelExact(N_qubits, bin_visible, N_samples, kernel_choice):
	#If the input corresponding to the kernel choice is either the gaussian kernel or the quantum kernel
	if (kernel_choice == 'Gaussian'):
		sigma = np.array([0.25, 10, 1000])
		k, k_exact_dict = GaussianKernelExact(N_qubits, bin_visible, sigma)
		#Gaussian approx kernel is equal to exact kernel
		k_exact = k
		k_dict = k_exact_dict
	elif (kernel_choice ==  'Quantum'):
		#compute for all binary strings
		ZZ, Z = EncodingFunc(N_qubits, bin_visible)
		k, k_exact, k_dict, k_exact_dict = QuantumKernelComputation(N_qubits, 2**N_qubits , 2**N_qubits, N_samples, ZZ, Z, ZZ, Z)
	else: raise IOError("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")

	#compute the expectation values for including each binary string sample
	return k, k_exact, k_dict, k_exact_dict


