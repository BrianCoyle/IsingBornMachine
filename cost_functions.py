import numpy as np
from random import *
from classical_kernel import GaussianKernelArray
from quantum_kernel import QuantumKernelArray
from kernel_functions import NormaliseKernel
from numpy import linalg as LA
from file_operations_in import KernelDictFromFile, DataDictFromFile
import stein_functions as sf
from auxiliary_functions import EmpiricalDist, SampleArrayToList, ToString
import sys
import json

def KernelSum(samplearray1, samplearray2, kernel_dict):
    '''
    This function computes the contribution to the MMD from the empirical distibutions
    from two sets of samples.
    kernel_dict contains the kernel values for all pairs of binary strings
    '''

    N_samples1 = samplearray1.shape[0]
    N_samples2 = samplearray2.shape[0]
    kernel_array = np.zeros((N_samples1, N_samples2)) 

    for sample1_index in range(0, N_samples1):
        for sample2_index in range(0, N_samples2):
            sample1 = ToString(samplearray1[sample1_index])
            sample2 = ToString(samplearray2[sample2_index])

            kernel_array[sample1_index, sample2_index] = kernel_dict[(sample1, sample2)]

    return kernel_array


def CostFunction(qc, cost_func, data_samples, data_exact_dict, born_samples, born_probs_dict,	
				N_samples, kernel_choice, stein_params, flag):
    '''
    This function computes the cost function between two distributions P and Q from samples from P and Q
    '''
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

                kernel_born_born_unnorm	,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, born_samples, 	born_samples)
                kernel_born_data_unnorm	,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, born_samples, 	data_samples)
                kernel_data_data_unnorm	,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, data_samples, 	data_samples)

        elif (flag == 'Precompute'):
            #Compute the empirical data distibution given samples

            kernel_dict  = KernelDictFromFile(qc, N_samples, kernel_choice)

            kernel_born_born_unnorm = KernelSum(born_samples, born_samples, kernel_dict)
            kernel_born_data_unnorm = KernelSum(born_samples, data_samples, kernel_dict)
            kernel_data_data_unnorm = KernelSum(data_samples, data_samples, kernel_dict)

        else: raise IOError('\'flag\' must be either \'Onfly\' or \'Precompute\'')

        loss =  NormaliseKernel(kernel_born_born_unnorm, 'same') - 2*NormaliseKernel(kernel_born_data_unnorm)+\
                    NormaliseKernel(kernel_data_data_unnorm, 'same')


    elif cost_func == 'Stein':

        sigma = np.array([0.25, 10, 1000])
        kernel_array = GaussianKernelArray(born_samples, born_samples, sigma)

        stein_flag = 'Onfly'
        kernel_stein_weighted = sf.WeightedKernel(qc,\
                                                    kernel_choice, kernel_array, N_samples, \
                                                    data_samples, data_exact_dict,     \
                                                    born_samples, born_samples,   \
                                                    stein_params, stein_flag, 'same')
    
        loss = NormaliseKernel(kernel_stein_weighted, 'same') 
    
    return loss

def CostGrad(qc, cost_func, data_samples, data_exact_dict,
			born_samples, born_probs_dict,
			born_samples_pm, 
			N_samples, kernel_choice, stein_params, flag):

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

                kernel_born_plus_unnorm	,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, born_samples, 	born_samples_plus)
                kernel_born_minus_unnorm,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, born_samples, 	born_samples_minus)
                kernel_data_plus_unnorm	,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, data_samples, 	born_samples_plus)
                kernel_data_minus_unnorm,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, data_samples, 	born_samples_minus)

      
        elif flag == 'Precompute':
				
            kernel_dict  = KernelDictFromFile(qc, N_samples, kernel_choice)

            kernel_born_plus_unnorm  = KernelSum(born_samples, born_samples_plus,    kernel_dict)
            kernel_born_minus_unnorm = KernelSum(born_samples, born_samples_minus,   kernel_dict)
            kernel_data_plus_unnorm  = KernelSum(data_samples, born_samples_plus,    kernel_dict)
            kernel_data_minus_unnorm = KernelSum(data_samples, born_samples_minus,   kernel_dict)

        else: raise IOError('\'flag\' must be either \'Onfly\' or \'Precompute\'')

        loss_grad = 2*(NormaliseKernel(kernel_born_minus_unnorm) - NormaliseKernel(kernel_born_plus_unnorm)-\
                    NormaliseKernel(kernel_data_minus_unnorm) + NormaliseKernel(kernel_data_plus_unnorm))

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
        kappa_q_born_bornplus = sf.WeightedKernel(qc, stein_kernel_choice, kernel_array_bornbornplus, N_samples, data_samples, data_exact_dict,\
                                                born_samples, born_samples_plus, stein_params, stein_flag)
        kappa_q_bornplus_born = sf.WeightedKernel(qc, stein_kernel_choice, kernel_array_bornbornplus, N_samples, data_samples, \
                                                data_exact_dict, born_samples_plus, born_samples, stein_params, stein_flag)
        kappa_q_born_bornminus = sf.WeightedKernel(qc, stein_kernel_choice, kernel_array_bornbornminus, N_samples, data_samples,\
                                                data_exact_dict, born_samples, born_samples_minus, stein_params, stein_flag)
        kappa_q_bornminus_born = sf.WeightedKernel(qc, stein_kernel_choice,kernel_array_bornbornminus, N_samples, data_samples,\
                                                data_exact_dict, born_samples_minus, born_samples, stein_params, stein_flag)

        loss_grad =   NormaliseKernel(kappa_q_born_bornminus) + NormaliseKernel(kappa_q_bornminus_born)\
                    - NormaliseKernel(kappa_q_born_bornplus) - NormaliseKernel(kappa_q_bornplus_born)
    else: raise IOError('\'cost_func\' must be either \'Stein\', or \'MMD\' ')
    return loss_grad

