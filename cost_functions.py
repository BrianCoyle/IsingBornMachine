import numpy as np
from random import *
from pyquil.api import get_qc
from classical_kernel import GaussianKernelArray
from quantum_kernel import QuantumKernelArray
from kernel_functions import NormaliseKernel

from file_operations_in import KernelDictFromFile, DataDictFromFile
import stein_functions as sf
from auxiliary_functions import EmpiricalDist, SampleArrayToList, FindNumQubits
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


def CostFunction(device_params, cost_func, data_samples, data_exact_dict, born_samples, born_probs_dict,	
				N_samples, kernel_choice, stein_params, flag):
    '''
    This function computes the cost function between two distributions P and Q from samples from P and Q
    '''
    N_qubits = FindNumQubits(device_params)
    if cost_func == 'MMD': 
        if (flag == 'Onfly'):
            if (kernel_choice == 'Gaussian'):
                sigma = np.array([0.25, 10, 1000])

                #Compute the Gaussian kernel on the fly for all samples in the sample space
                kernel_born_born_unnorm 	= GaussianKernelArray(born_samples, born_samples, 	sigma)
                kernel_born_data_unnorm 	= GaussianKernelArray(born_samples, data_samples, 	sigma)	
                kernel_data_data_unnorm 	= GaussianKernelArray(data_samples, data_samples, 	sigma)

            elif kernel_choice == 'Quantum':
                N_kernel_samples = N_samples[-1] #Number of kernel samples is the last element of N_samples
                #Compute the Quantum kernel on the fly for all pairs of samples required

                kernel_born_born_unnorm	,_,_,_ = QuantumKernelArray(device_params, N_kernel_samples, born_samples, 	born_samples)
                kernel_born_data_unnorm	,_,_,_ = QuantumKernelArray(device_params, N_kernel_samples, born_samples, 	data_samples)
                kernel_data_data_unnorm	,_,_,_ = QuantumKernelArray(device_params, N_kernel_samples, data_samples, 	data_samples)

            loss =  NormaliseKernel(kernel_born_born_unnorm, 'same') - 2*NormaliseKernel(kernel_born_data_unnorm)+\
                    NormaliseKernel(kernel_data_data_unnorm, 'same')

        elif (flag == 'Precompute'):
            #Compute the empirical data distibution given samples

            kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_samples, kernel_choice)

            k_bb= KernelSum(N_qubits, born_samples, born_samples, kernel_sampled_dict,'same')
            k_bd = KernelSum(N_qubits, born_samples, data_samples, kernel_sampled_dict)
            k_dd = KernelSum(N_qubits, data_samples, data_samples, kernel_sampled_dict, 'same')

            loss =  k_bb - 2*k_bd + k_dd

        else: raise IOError('\'flag\' must be either \'Onfly\' or \'Precompute\'')
    elif cost_func == 'Stein':

        sigma = np.array([0.25, 10, 1000])
        kernel_array = GaussianKernelArray(born_samples, born_samples, sigma)

        stein_flag = 'Onfly'
        kernel_stein_weighted = sf.WeightedKernel(device_params,\
                                                    kernel_choice, kernel_array, N_samples, \
                                                    data_samples, data_exact_dict,     \
                                                    born_samples, born_samples,   \
                                                    stein_params, stein_flag, 'same')
    
        loss = NormaliseKernel(kernel_stein_weighted, 'same') 
    
    return loss

def CostGrad(device_params, cost_func, data_samples, data_exact_dict,
			born_samples, born_probs_dict,
			born_samples_pm, 
			N_samples, kernel_choice, stein_params, flag):

    N_qubits = FindNumQubits(device_params)
    [born_samples_plus, born_samples_minus] = born_samples_pm

    if cost_func == 'MMD':
        if flag == 'Onfly':
            if kernel_choice == 'Gaussian':
                sigma = np.array([0.25, 10, 1000])
                #Compute the Gaussian kernel on the fly for all pairs of samples required
                kernel_born_plus_unnorm 	= GaussianKernelArray(born_samples, born_samples_plus, 	sigma)
                kernel_born_minus_unnorm 	= GaussianKernelArray(born_samples, born_samples_minus, sigma)
                kernel_data_plus_unnorm 	= GaussianKernelArray(data_samples, born_samples_plus, 	sigma)
                kernel_data_minus_unnorm 	= GaussianKernelArray(data_samples, born_samples_minus, sigma)

            elif kernel_choice == 'Quantum':
                N_kernel_samples = N_samples[-1] #Number of kernel samples is the last element of N_samples
                #Compute the Quantum kernel on the fly for all pairs of samples required

                kernel_born_plus_unnorm	,_,_,_ = QuantumKernelArray(device_params, N_kernel_samples, born_samples, 	born_samples_plus)
                kernel_born_minus_unnorm,_,_,_ = QuantumKernelArray(device_params, N_kernel_samples, born_samples, 	born_samples_minus)
                kernel_data_plus_unnorm	,_,_,_ = QuantumKernelArray(device_params, N_kernel_samples, data_samples, 	born_samples_plus)
                kernel_born_minus_unnorm,_,_,_ = QuantumKernelArray(device_params, N_kernel_samples, data_samples, 	born_samples_minus)

            loss_grad = 2*(NormaliseKernel(kernel_born_minus_unnorm) - NormaliseKernel(kernel_born_plus_unnorm)-\
                            NormaliseKernel(kernel_data_minus_unnorm) + NormaliseKernel(kernel_data_plus_unnorm))

        elif flag == 'Precompute':
				
            kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_samples, kernel_choice)

            k_bornplus_born = KernelSum(N_qubits,   born_samples, born_samples_plus,    kernel_sampled_dict)
            k_bornminus_born = KernelSum(N_qubits,  born_samples, born_samples_minus,   kernel_sampled_dict)
            k_bornplus_data = KernelSum(N_qubits,   data_samples, born_samples_plus,    kernel_sampled_dict)
            k_bornminus_data = KernelSum(N_qubits,  data_samples, born_samples_minus,   kernel_sampled_dict)

            loss_grad =  2*(k_bornminus_born- k_bornplus_born - k_bornminus_data + k_bornplus_data)

    elif cost_func == 'Stein':
        if kernel_choice == 'Gaussian':
            sigma = np.array([0.25, 10, 1000])
            born_samples_plus = born_samples_pm[0]
            born_samples_minus = born_samples_pm[1]
            kernel_array_bornbornplus = GaussianKernelArray(born_samples, 	born_samples_plus, sigma)
            kernel_array_bornbornminus = GaussianKernelArray(born_samples, 	born_samples_minus, sigma)

        stein_flag = 'Onfly'
        stein_kernel_choice = stein_params[3]

        # Compute the weighted kernel for each pair of samples required in the gradient of Stein Cost Function
        kappa_q_born_bornplus = sf.WeightedKernel(device_params, stein_kernel_choice, kernel_array_bornbornplus, N_samples, data_samples, data_exact_dict,\
                                                born_samples, born_samples_plus, stein_params, stein_flag)
        kappa_q_bornplus_born = sf.WeightedKernel(device_params, stein_kernel_choice, kernel_array_bornbornplus, N_samples, data_samples, \
                                                data_exact_dict, born_samples_plus, born_samples, stein_params, stein_flag)
        kappa_q_born_bornminus = sf.WeightedKernel(device_params, stein_kernel_choice, kernel_array_bornbornminus, N_samples, data_samples,\
                                                data_exact_dict, born_samples, born_samples_minus, stein_params, stein_flag)
        kappa_q_bornminus_born = sf.WeightedKernel(device_params, stein_kernel_choice,kernel_array_bornbornminus, N_samples, data_samples,\
                                                data_exact_dict, born_samples_minus, born_samples, stein_params, stein_flag)

        loss_grad =   NormaliseKernel(kappa_q_born_bornminus) + NormaliseKernel(kappa_q_bornminus_born)\
                    - NormaliseKernel(kappa_q_born_bornplus) - NormaliseKernel(kappa_q_bornplus_born)
    else: raise IOError('\'cost_func\' must be either \'Stein\', or \'MMD\' ')
    return loss_grad

