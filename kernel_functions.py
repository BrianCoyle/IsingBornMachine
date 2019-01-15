from pyquil.quil import Program
import numpy as np
from pyquil.api import get_qc

from classical_kernel import GaussianKernelArray, GaussianKernelDict
from quantum_kernel import QuantumKernelArray

from auxiliary_functions import AllBinaryStrings, FindNumQubits

def KernelAllBinaryStrings(device_params, N_samples, kernel_choice):
    '''
    This functions computes the kernel, either Gaussian or Quantum for *all* Binary strings of 
    length N_qubits
    '''
    N_qubits = FindNumQubits(device_params)
    binary_strings_array = AllBinaryStrings(N_qubits)
    if (kernel_choice == 'Gaussian'):
        sigma = np.array([0.25, 10, 1000])
        kernel_approx_array = GaussianKernelArray(binary_strings_array, binary_strings_array, sigma)
        #Gaussian approx kernel is equal to exact kernel
        kernel_exact_array = kernel_approx_array
        kernel_exact_dict = GaussianKernelDict(binary_strings_array, binary_strings_array, sigma)
        kernel_approx_dict = kernel_exact_dict
    elif (kernel_choice ==  'Quantum'):
        #compute for all binary strings
        kernel_approx_array, kernel_exact_array, kernel_approx_dict, kernel_exact_dict =\
            QuantumKernelArray(device_params, N_samples, binary_strings_array, binary_strings_array)
    else: raise IOError("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")

    #compute the expectation values for including each binary string sample
    return kernel_approx_array, kernel_exact_array, kernel_approx_dict, kernel_exact_dict

def NormaliseKernel(kernel_array, *argsv):
    '''
    This function sums and normalises the kernel matrix to be used in the 
    training cost function
    '''
    if type(kernel_array) is not np.ndarray:
        raise IOError('\'kernel_array\' must be a np.ndarray')

    [N_samples1, N_samples2] = kernel_array.shape
    if ('same' not in argsv):
        #if the two sets of samples come from the same distribution
        normed_kernel = (1/(N_samples1*(N_samples2)))*kernel_array.sum()
    else:
        normed_kernel = (1/(N_samples1*(N_samples2 - 1)))*kernel_array.sum()
    
    return normed_kernel