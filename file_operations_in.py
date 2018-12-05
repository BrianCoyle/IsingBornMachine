## @package file_operations_in import functions 
#
# A collection of functions for imported pre-computed data

import numpy as np
import ast
import sys
import json
from auxiliary_functions import SampleListToArray

def FileLoad(file):
	kernel = json.load(file)
	kernel_dict = json.loads(kernel)
	dict_keys = kernel_dict.keys()
	dict_values = kernel_dict.values()
	k1 = [eval(key) for key in dict_keys]
	return kernel_dict, k1, dict_values

## Reads data dictionary from file
#
# @param[in] N_qubits The number of qubits
# @param[in] N_samples The number of samples
#
# @returns A dictionary containing the appropriate data
def DataDictFromFile(N_qubits, N_samples):
	print(N_samples)
	if (N_samples == 'infinite'):
		with open('Data_Dict_Exact_%iQBs' % N_qubits, 'r') as f:
			raw_from_file = json.load(f)
			data_dict = json.loads(raw_from_file)
	else: 
		print(N_samples)
		with open('Data_Dict_%iSamples_%iQBs' % (N_samples[0], N_qubits), 'r') as g:
			raw_from_file = json.load(g)
			data_dict = json.loads(raw_from_file)
	return data_dict

## Returns relevant data
#
# @param[in] approx The approximation type
# @param[in] N_qubits The number of qubits
# @param[in] N_data_samples The number of data samples
# @param[in] stein_approx The approximation type
#
# @param[out] data_samples The requested list of samples
# @param[out] data_exact_dict The requested dictionary of exact samples
#
# @return Requested data
def DataImport(approx, N_qubits, N_data_samples, stein_approx):
    
    data_exact_dict = DataDictFromFile(N_qubits, 'infinite')
    
    if (approx == 'Sampler'):
        
        data_samples_orig = list(np.loadtxt('Data_%iQBs_%iSamples' % (N_qubits, N_data_samples), dtype = str))
        data_samples = SampleListToArray(data_samples_orig, N_qubits)
    
    elif (approx == 'Exact') or (stein_approx == 'Exact_Stein'):
        
        data_samples = []
    
    else: raise IOError('Please enter either \'Sampler\' or \'Exact\' for \'approx\' ')
    
    return data_samples, data_exact_dict 

## Reads kernel dictionary from file
def KernelDictFromFile(N_qubits, N_samples, kernel_choice):
	
    #reads kernel dictionary from file
    N_kernel_samples = N_samples[3]

    if (N_kernel_samples == 'infinite'):
        with open('%sKernel_Exact_Dict_%iQBs' % (kernel_choice[0], N_qubits), 'r') as f:
            kernel_dict, k1, v = FileLoad(f)
    else:
        with open('%sKernel_Dict_%iQBs_%iKernelSamples' % (kernel_choice[0], N_qubits,N_kernel_samples), 'r') as f:
            kernel_dict, k1, v = FileLoad(f)

    return dict(zip(*[k1,v]))


def ParamsFromFile(N_qubits):
	Params = np.load('Parameters_%iQubits.npz' % (N_qubits))
	J_i = Params['J_init']
	b_i = Params['b_init']
	g_x_i = Params['gamma_x_init']
	g_y_i = Params['gamma_y_init']
	
	return J_i, b_i, g_x_i, g_y_i
