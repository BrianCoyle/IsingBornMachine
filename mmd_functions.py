from pyquil.quil import Program
from pyquil.paulis import *
from pyquil.gates import *
import numpy as np
from pyquil.api import QVMConnection
from random import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin
from param_init import StateInit, NetworkParams

from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler
from classical_kernel import GaussianKernel, GaussianKernelExact
from file_operations_in import KernelDictFromFile, DataDictFromFile
from quantum_kernel import KernelCircuit, QuantumKernelComputation, EncodingFunc

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
	
def KernelSum(N_v, samplearray1, samplearray2,
			kernel_sampled_dict, *argsv):
	'''This function computes the contribution to the MMD from the empirical distibutions
			from two sets of samples'''

	N_sample1 = samplearray1.shape[0]
	N_sample2 = samplearray2.shape[0]

	empiricaldist_dict1 = EmpiricalDist(samplearray1, N_v)
	empiricaldist_dict2 = EmpiricalDist(samplearray2, N_v)
	
	samples_list_1 = SampleArrayToList(samplearray1)
	samples_list_2 = SampleArrayToList(samplearray2)

	kernel_sum_unnorm = 0
	for sample1 in samples_list_1:
		for sample2 in  samples_list_2:
			if (sample1 == sample2):
				# if ('same' not in argsv):
					kernel_sum_unnorm = kernel_sum_unnorm + kernel_sampled_dict[sample1, sample2]
			else:
				kernel_sum_unnorm = kernel_sum_unnorm + kernel_sampled_dict[sample1, sample2]
	
	if ('same' not in argsv):
		kernel_sum = (1/(N_sample1*N_sample2))*kernel_sum_unnorm
	else: 
		kernel_sum = (1/(N_sample1*(N_sample2-1)))*kernel_sum_unnorm

	kernel_prob = SumContribution(N_v, empiricaldist_dict1 ,empiricaldist_dict2, kernel_sampled_dict)


	return kernel_sum, kernel_prob

def MMDGrad(N_v, data, data_exact_dict,
						born_samples, born_probs_dict,
						born_samples_plus, born_plus_exact_dict, 
						born_samples_minus, born_minus_exact_dict,
						N_k_samples, k_choice, approx):


	if (approx == 'Sampler'):

		kernel_sampled_dict  = KernelDictFromFile(N_v, N_k_samples, k_choice)

		k_bornplus_born, born_born_plus = KernelSum(N_v, born_samples, born_samples_plus,\
														kernel_sampled_dict)
		k_bornminus_born, born_born_minus = KernelSum(N_v, born_samples, born_samples_minus,\
														kernel_sampled_dict)
		# print(empiricaldist_dicta,'\n', empiricaldist_dict1)
		k_bornplus_data,  data_born_plus= KernelSum(N_v, data, born_samples_plus,\
														kernel_sampled_dict)
		k_bornminus_data,  data_born_minus = KernelSum(N_v, data, born_samples_minus,\
														kernel_sampled_dict)
		
		# print(bornplus_born)
		# print(bornminus_born)
		#L1_MMD_Grad = 2*(k_bornminus_born - k_bornplus_born- k_bornminus_data + k_bornplus_data)

	elif (approx == 'Exact'):
		#We have dictionaries of 5 files, the kernel, the data, the born probs, the born probs plus/minus
		kernel_dict = KernelDictFromFile(N_v, N_k_samples, k_choice)
		#compute the term kp_1p_2 to add to expectation value for each pair, from bornplus/bornminus, with born machine and data
		born_born_plus = SumContribution(N_v, born_probs_dict, born_plus_exact_dict, kernel_dict)
		born_born_minus = SumContribution(N_v, born_probs_dict, born_minus_exact_dict, kernel_dict)
		data_born_plus = SumContribution(N_v, data_exact_dict, born_plus_exact_dict, kernel_dict)
		data_born_minus = SumContribution(N_v, data_exact_dict, born_minus_exact_dict, kernel_dict)

		#Return the gradient of the loss function (L = MMD^2) for a given parameter
	L_MMD_Grad = 2*(born_born_minus.sum()-born_born_plus.sum()-data_born_minus.sum()+data_born_plus.sum())
	return L_MMD_Grad



def MMDCost(N_v, data, data_exact_dict, born_samples, born_probs_dict,	
			N_kernel_samples, kernel_choice,
			approx):
	'''This function computes the MMD cost function between P and Q from samples from P and
		Q if the sampling approx is taken, otherwise if the cost function is to be computed exactly'''

	if (approx == 'Sampler'):
		#Compute the empirical data distibution given samples
	

		kernel_sampled_dict  = KernelDictFromFile(N_v, N_kernel_samples, kernel_choice)

		k_bb, born_born = KernelSum(N_v, born_samples, born_samples, kernel_sampled_dict, 'same')
		k_bd, born_data = KernelSum(N_v, born_samples, data, kernel_sampled_dict)
		k_dd, data_data = KernelSum(N_v, data, data, kernel_sampled_dict,  'same')
		

		#L1 = k_bb + k_dd - 2*(k_bd)
		L_mmd =  born_born.sum() - 2*born_data.sum() + data_data.sum()

	elif (approx == 'Exact'):

		#compute the term kp_1p_2 to add to expectation value for each pair, from born/bornplus/bornminus machine, and data
		k_exact_dict = KernelDictFromFile(N_v, 'infinite', kernel_choice)

		born_born = SumContribution(N_v, born_probs_dict, born_probs_dict, k_exact_dict,'same')
		born_data = SumContribution(N_v, data_exact_dict, born_probs_dict, k_exact_dict)
		data_data = SumContribution(N_v, data_exact_dict, data_exact_dict, k_exact_dict, 'same')
		# print(born_born)
		# print('\n', data_data, '\n', born_data)
		#return the loss function (L = MMD^2)
		L_mmd = born_born.sum() - 2*born_data.sum() + data_data.sum()

	return L_mmd


def MMDKernel(N_v, bin_visible, N_k_samples, k_choice):
	#If the input corresponding to the kernel choice is either the gaussian kernel or the quantum kernel
	if (k_choice == 'Gaussian'):
		sigma = np.array([0.25, 10, 1000])
		k, k_exact_dict = GaussianKernelExact(N_v, bin_visible, sigma)
		#Gaussian approx kernel is equal to exact kernel
		k_exact = k
		k_dict = k_exact_dict
	elif (k_choice ==  'Quantum'):
		#compute for all binary strings
		ZZ, Z = EncodingFunc(N_v, bin_visible)
		k, k_exact, k_dict, k_exact_dict = QuantumKernelComputation(N_v, 2**N_v , 2**N_v, N_k_samples, ZZ, Z, ZZ, Z)
	else: raise IOError("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")

	#compute the expectation values for including each binary string sample
	return k, k_exact, k_dict, k_exact_dict


