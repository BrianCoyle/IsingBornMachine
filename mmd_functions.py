import numpy as np
from random import *
from pyquil.api import get_qc
from classical_kernel import GaussianKernelArray, NormaliseKernel
from file_operations_in import KernelDictFromFile, DataDictFromFile

from auxiliary_functions import ConvertToString, EmpiricalDist, SampleArrayToList
import sys
import json

def KernelSum(N_qubits, samplearray1, samplearray2, kernel_sampled_dict, *argsv):
	'''This function computes the contribution to the MMD from the empirical distibutions
			from two sets of samples'''

	N_samples1 = samplearray1.shape[0]
	N_samples2 = samplearray2.shape[0]

	samples_list_1 = SampleArrayToList(samplearray1)
	samples_list_2 = SampleArrayToList(samplearray2)
	kernel_sum = 0
	i = 0
	j = 0
	for sample1 in samples_list_1:
		i += 1
		for sample2 in  samples_list_2:
			j += 1
			if ('same' in argsv):
				#If the term in the cost function or gradient involves the same distribution,
				#don't include the kernel values comparing the same sample in the sum
				if (i != j ):
					kernel_sum = kernel_sum + (1/(N_samples1*(N_samples2-1)))*kernel_sampled_dict[sample1, sample2]
			else:
				kernel_sum = kernel_sum + (1/(N_samples1*N_samples2))*kernel_sampled_dict[sample1, sample2]
			# else:
			# 	kernel_sum_unnorm = kernel_sum_unnorm + kernel_sampled_dict[sample1, sample2]
	
	return kernel_sum

def MMDGrad(device_params, data, data_exact_dict,
			born_samples, born_probs_dict,
			born_samples_plus,  
			born_samples_minus, 
			N_samples, kernel_choice, flag):
	device_name = device_params[0]
	as_qvm_value = device_params[1]

	qc = get_qc(device_name, as_qvm = as_qvm_value)
	qubits = qc.qubits()
	N_qubits = len(qubits)
	if (flag == 'Onfly'):
		if (kernel_choice == 'Gaussian'):
			sigma = np.array([0.25, 10, 1000])

			#Compute the Gaussian kernel on the fly for all pairs of samples required
			gaussian_born_plus = NormaliseKernel(GaussianKernelArray(born_samples, born_samples_plus, sigma))	
			gaussian_born_minus = NormaliseKernel(GaussianKernelArray(born_samples, born_samples_minus, sigma))	
			gaussian_data_plus = NormaliseKernel(GaussianKernelArray(data, born_samples_plus, sigma))
			gaussian_data_minus = NormaliseKernel(GaussianKernelArray(data, born_samples_minus, sigma))

			L_MMD_Grad = 2*(gaussian_born_minus - gaussian_born_plus-gaussian_data_minus + gaussian_data_plus)

	elif (flag == 'Precompute'):
			
		kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_samples, kernel_choice)

		k_bornplus_born = KernelSum(N_qubits, born_samples, born_samples_plus,kernel_sampled_dict)
		k_bornminus_born = KernelSum(N_qubits, born_samples, born_samples_minus,kernel_sampled_dict)
		k_bornplus_data = KernelSum(N_qubits, data, born_samples_plus,kernel_sampled_dict)
		k_bornminus_data = KernelSum(N_qubits, data, born_samples_minus,kernel_sampled_dict)

		L_MMD_Grad =  2*(k_bornminus_born- k_bornplus_born- k_bornminus_data + k_bornplus_data)

	return L_MMD_Grad


def MMDCost(device_params, data, data_exact_dict, born_samples, born_probs_dict,	
			N_samples, kernel_choice, flag):

	'''This function computes the MMD cost function between P and Q from samples from P and
		Q if the sampling approx is taken, otherwise if the cost function is to be computed exactly'''
	device_name = device_params[0]
	as_qvm_value = device_params[1]

	qc = get_qc(device_name, as_qvm = as_qvm_value)
	qubits = qc.qubits()
	N_qubits = len(qubits)

	if (flag == 'Onfly'):
		if (kernel_choice == 'Gaussian'):

			sigma = np.array([0.25, 10, 1000])

			#Compute the Gaussian kernel on the fly for all samples in the sample space
			gaussian_born_born = NormaliseKernel(GaussianKernelArray(born_samples, born_samples, sigma),'same')
			gaussian_born_data = NormaliseKernel(GaussianKernelArray(born_samples, data, sigma))
			gaussian_data_data = NormaliseKernel(GaussianKernelArray(data, data, sigma), 'same')	

			L_mmd =  gaussian_born_born - 2*gaussian_born_data + gaussian_data_data

	elif (flag == 'Precompute'):
		#Compute the empirical data distibution given samples
	
		kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_samples, kernel_choice)

		k_bb= KernelSum(N_qubits, born_samples, born_samples, kernel_sampled_dict,'same')
		k_bd = KernelSum(N_qubits, born_samples, data, kernel_sampled_dict)
		k_dd = KernelSum(N_qubits, data, data, kernel_sampled_dict, 'same')

		L_mmd =  k_bb - 2*k_bd + k_dd

	else: raise IOError('\'flag\' must be either \'Onfly\' or \'Precompute\'')
	return L_mmd





