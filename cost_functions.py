import numpy as np
from random import *
from classical_kernel import GaussianKernelArray
from quantum_kernel import QuantumKernelArray
from numpy import linalg as LA
from file_operations_in import KernelDictFromFile
import stein_functions_temp as sf
import sinkhorn_functions as shornfun
import auxiliary_functions as aux
import sys
import json
import time

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
            sample1 = aux.ToString(samplearray1[sample1_index])
            sample2 = aux.ToString(samplearray2[sample2_index])
            kernel_array[sample1_index, sample2_index] = kernel_dict[(sample1, sample2)]

    return kernel_array


def CostFunction(qc, cost_func, data_samples, data_exact_dict, born_samples, born_probs_dict,	
				N_samples, kernel_choice, stein_params, flag, sinkhorn_eps):
    '''
    This function computes the cost function between two distributions P and Q from samples from P and Q
    '''

    #Extract unique samples and corresponding empirical probabilities from set of samples
    born_emp_samples, born_emp_probs, _, _ = aux.ExtractSampleInformation(born_samples)
    data_emp_samples, data_emp_probs, _, _ = aux.ExtractSampleInformation(data_samples)
    if cost_func.lower() == 'mmd': 
                
        if (flag.lower() == 'onfly'):
            if (kernel_choice.lower() == 'gaussian'):
                sigma = np.array([0.25, 10, 1000])
                #Compute the Gaussian kernel on the fly for all samples in the sample space
                kernel_born_born_emp 	= GaussianKernelArray(born_emp_samples, born_emp_samples, 	sigma)
                kernel_born_data_emp 	= GaussianKernelArray(born_emp_samples, data_emp_samples, 	sigma)	
                kernel_data_data_emp 	= GaussianKernelArray(data_emp_samples, data_emp_samples, 	sigma)

            elif kernel_choice.lower() == 'quantum':
                N_kernel_samples = N_samples[-1] #Number of kernel samples is the last element of N_samples
                #Compute the Quantum kernel on the fly for all pairs of samples required

                kernel_born_born_emp	,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, born_emp_samples, 	born_emp_samples)
                kernel_born_data_emp	,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, born_emp_samples, 	data_emp_samples)
                kernel_data_data_emp	,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, data_emp_samples, 	data_emp_samples)

        elif (flag.lower() == 'precompute'):
            #Compute the empirical data distibution given samples

            kernel_dict  = KernelDictFromFile(qc, N_samples, kernel_choice)

            kernel_born_born_emp = KernelSum(born_emp_samples, born_emp_samples, kernel_dict)
            kernel_born_data_emp = KernelSum(born_emp_samples, data_emp_samples, kernel_dict)
            kernel_data_data_emp = KernelSum(data_emp_samples, data_emp_samples, kernel_dict)

        else: raise ValueError('\'flag\' must be either \'Onfly\' or \'Precompute\'')

    
        loss    =  np.dot(np.dot(born_emp_probs, kernel_born_born_emp), born_emp_probs) \
                -  2*np.dot(np.dot(born_emp_probs, kernel_born_data_emp), data_emp_probs) \
                +  np.dot(np.dot(data_emp_probs, kernel_data_data_emp), data_emp_probs) 

    elif cost_func.lower() == 'stein':

        if flag.lower() == 'onfly':
            if (kernel_choice.lower() == 'gaussian'):

                sigma = np.array([0.25, 10, 1000])
                kernel_array = GaussianKernelArray(born_emp_samples, born_emp_samples, sigma)

            elif kernel_choice.lower() == 'quantum':
                kernel_array ,_,_,_ = QuantumKernelArray(qc, N_kernel_samples, born_samples, born_samples)

            else: raise ValueError('Stein only supports Gaussian kernel currently')
        elif flag.lower() == 'precompute':

            kernel_dict  = KernelDictFromFile(qc, N_samples, kernel_choice)

            kernel_array = KernelSum(born_emp_samples, born_emp_samples, kernel_dict)

        else: raise ValueError('\'flag\' must be either \'Onfly\' or \'Precompute\'')

        stein_flag = 'Precompute'
        kernel_stein_weighted = sf.WeightedKernel(qc,kernel_choice, kernel_array, N_samples,    \
                                                data_samples, data_exact_dict,                  \
                                                born_emp_samples, born_emp_samples,             \
                                                stein_params, stein_flag)
    

        loss = np.dot(np.dot(born_emp_probs, kernel_stein_weighted), born_emp_probs)

    elif cost_func.lower() == 'sinkhorn':
        #If Sinkhorn cost function to be used
        loss = shornfun.FeydySink(born_samples, data_samples, sinkhorn_eps).item()
        
    else: raise ValueError('\'cost_func\' must be either \'MMD\', \'Stein\', or \'Sinkhorn\' ')

    return loss


def CostGrad(qc, cost_func, data_samples, data_exact_dict,
			born_samples, born_probs_dict, born_samples_pm, 
			N_samples, kernel_choice, stein_params, flag, sinkhorn_eps):

    [born_samples_plus, born_samples_minus] = born_samples_pm

    born_emp_samples, born_emp_probs, _, _              = aux.ExtractSampleInformation(born_samples)
    data_emp_samples, data_emp_probs, _, _              = aux.ExtractSampleInformation(data_samples)
    born_plus_emp_samples, born_plus_emp_probs, _, _    = aux.ExtractSampleInformation(born_samples_plus)
    born_minus_emp_samples, born_minus_emp_probs, _, _  = aux.ExtractSampleInformation(born_samples_minus)

    if cost_func == 'MMD':

        if flag == 'Onfly':
            if kernel_choice == 'Gaussian':
                sigma = np.array([0.25, 10, 1000])
                #Compute the Gaussian kernel on the fly for all pairs of samples required
                kernel_born_plus_emp 	= GaussianKernelArray(born_emp_samples, born_plus_emp_samples, 	sigma)
                kernel_born_minus_emp 	= GaussianKernelArray(born_emp_samples, born_minus_emp_probs, sigma)
                kernel_data_plus_emp 	= GaussianKernelArray(data_emp_samples, born_plus_emp_samples, 	sigma)
                kernel_data_minus_emp 	= GaussianKernelArray(data_emp_samples, born_minus_emp_probs, sigma)

            elif kernel_choice == 'Quantum':
                N_kernel_samples = N_samples[-1] #Number of kernel samples is the last element of N_samples
                #Compute the Quantum kernel on the fly for all pairs of samples required

                kernel_born_plus_emp ,_,_,_     = QuantumKernelArray(qc, N_kernel_samples, born_emp_samples, 	born_plus_emp_samples)
                kernel_born_minus_emp,_,_,_     = QuantumKernelArray(qc, N_kernel_samples, born_emp_samples, 	born_minus_emp_probs)
                kernel_data_plus_emp ,_,_,_     = QuantumKernelArray(qc, N_kernel_samples, data_emp_samples, 	born_plus_emp_samples)
                kernel_data_minus_emp,_,_,_     = QuantumKernelArray(qc, N_kernel_samples, data_emp_samples, 	born_minus_emp_probs)

      
        elif flag == 'Precompute':
			#To speed up computation, read in precomputed kernel dicrionary from a file.
            kernel_dict  = KernelDictFromFile(qc, N_samples, kernel_choice)

            kernel_born_plus_emp    = KernelSum(born_emp_samples, born_plus_emp_samples, kernel_dict)
            kernel_born_minus_emp   = KernelSum(born_emp_samples, born_minus_emp_samples, kernel_dict)
            kernel_data_plus_emp    = KernelSum(data_emp_samples, born_plus_emp_samples, kernel_dict)
            kernel_data_minus_emp   = KernelSum(data_emp_samples, born_minus_emp_samples, kernel_dict)

    
        else: raise ValueError('\'flag\' must be either \'Onfly\' or \'Precompute\'')

        loss_grad = 2*(  np.dot(np.dot(born_emp_probs, kernel_born_minus_emp), born_minus_emp_probs)    \
                        - np.dot(np.dot(born_emp_probs, kernel_born_plus_emp), born_plus_emp_probs)     \
                        - np.dot(np.dot(data_emp_probs, kernel_data_minus_emp), born_minus_emp_probs)   \
                        + np.dot(np.dot(data_emp_probs, kernel_data_plus_emp), born_plus_emp_probs)     )      

    elif cost_func == 'Stein':

        sigma = np.array([0.25, 10, 1000])
        [born_samples_plus, born_samples_minus] = born_samples_pm

    
        if flag == 'Onfly':
            if kernel_choice == 'Gaussian':
                sigma = np.array([0.25, 10, 1000])
                #Compute the Gaussian kernel on the fly for all pairs of samples required
                kernel_born_plus_emp 	= GaussianKernelArray(born_emp_samples, born_plus_emp_samples, 	sigma)
                kernel_born_minus_emp 	= GaussianKernelArray(born_emp_samples, born_minus_emp_probs, sigma)
           
            elif kernel_choice == 'Quantum':
                N_kernel_samples = N_samples[-1] #Number of kernel samples is the last element of N_samples
                #Compute the Quantum kernel on the fly for all pairs of samples required

                kernel_born_plus_emp ,_,_,_     = QuantumKernelArray(qc, N_kernel_samples, born_emp_samples, 	born_plus_emp_samples)
                kernel_born_minus_emp,_,_,_     = QuantumKernelArray(qc, N_kernel_samples, born_emp_samples, 	born_minus_emp_probs)
      
        elif flag == 'Precompute':
			#To speed up computation, read in precomputed kernel dicrionary from a file.
            kernel_dict  = KernelDictFromFile(qc, N_samples, kernel_choice)

            kernel_born_plus_emp    = KernelSum(born_emp_samples, born_plus_emp_samples, kernel_dict)
            kernel_born_minus_emp   = KernelSum(born_emp_samples, born_minus_emp_samples, kernel_dict)

        kernel_plus_born_emp    = np.transpose(kernel_born_plus_emp)
        kernel_minus_born_emp   = np.transpose(kernel_born_minus_emp)

        stein_kernel_choice = stein_params[3]

        # Compute the weighted kernel for each pair of samples required in the gradient of Stein Cost Function
        kappa_q_born_bornplus = sf.WeightedKernel(qc, stein_kernel_choice, kernel_born_plus_emp, N_samples, data_samples, data_exact_dict,\
                                                born_emp_samples, born_plus_emp_samples, stein_params, flag)
        kappa_q_bornplus_born = sf.WeightedKernel(qc, stein_kernel_choice, kernel_plus_born_emp, N_samples, data_samples, \
                                                data_exact_dict, born_plus_emp_samples, born_emp_samples, stein_params, flag)
        kappa_q_born_bornminus = sf.WeightedKernel(qc, stein_kernel_choice, kernel_born_minus_emp, N_samples, data_samples,\
                                                data_exact_dict, born_emp_samples, born_minus_emp_samples, stein_params, flag)
        kappa_q_bornminus_born = sf.WeightedKernel(qc, stein_kernel_choice, kernel_minus_born_emp, N_samples, data_samples,\
                                                data_exact_dict, born_minus_emp_samples, born_emp_samples, stein_params, flag)


        loss_grad = np.dot(np.dot(born_emp_probs, kappa_q_born_bornminus), born_minus_emp_probs)    \
                        + np.dot(np.dot(born_minus_emp_probs, kappa_q_bornminus_born), born_emp_probs)     \
                        - np.dot(np.dot(born_emp_probs, kappa_q_born_bornplus), born_plus_emp_probs)   \
                        - np.dot(np.dot(born_plus_emp_probs, kappa_q_bornplus_born), born_emp_probs)          
      
    
    elif cost_func == 'Sinkhorn':
        # loss_grad = shornfun.SinkhornGrad(born_samples_pm, data_samples, sinkhorn_eps)
        loss_grad = shornfun.SinkGrad(born_samples, born_samples_pm, data_samples, sinkhorn_eps)

    else: raise ValueError('\'cost_func\' must be either \'MMD\', \'Stein\', or \'Sinkhorn\' ')

    return loss_grad

