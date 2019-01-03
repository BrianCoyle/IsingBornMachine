from pyquil.quil import Program
import numpy as np
from pyquil.api import get_qc

from classical_kernel import GaussianKernelArray, GaussianKernelDict
from quantum_kernel import QuantumKernelComputation

def KernelAllBinaryStrings(device_params, binary_strings_array, N_samples, kernel_choice):
	#If the input corresponding to the kernel choice is either the gaussian kernel or the quantum kernel
    device_name = device_params[0]
    as_qvm_value = device_params[1]

    qc = get_qc(device_name, as_qvm = as_qvm_value)
    qubits = qc.qubits()
    N_qubits =len(qubits)
    if (kernel_choice == 'Gaussian'):
        print(binary_strings_array)
        sigma = np.array([0.25, 10, 1000])
        kernel_approx_array = GaussianKernelArray(binary_strings_array, binary_strings_array, sigma)
        #Gaussian approx kernel is equal to exact kernel
        kernel_exact_array = kernel_approx_array
        kernel_exact_dict = GaussianKernelDict(binary_strings_array, binary_strings_array, sigma)
        kernel_approx_dict = kernel_exact_dict
    elif (kernel_choice ==  'Quantum'):
        #compute for all binary strings
        kernel_approx_array, kernel_exact_array, kernel_approx_dict, kernel_exact_dict =\
        QuantumKernelComputation(device_params, 2**N_qubits , 2**N_qubits, N_samples, binary_strings_array, binary_strings_array)
    else: raise IOError("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")

    #compute the expectation values for including each binary string sample
    return kernel_approx_array, kernel_exact_array, kernel_approx_dict, kernel_exact_dict