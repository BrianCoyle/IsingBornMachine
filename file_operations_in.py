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

def DataDictFromFile(data_type, N_qubits, N_samples, *args):
	if data_type == 'Classical_Data':
		if (N_samples == 'infinite'):
			with open('data/Classical_Data_Dict_%iQBs_Exact' % (N_qubits), 'r') as f:
				raw_from_file = json.load(f)
				data_dict = json.loads(raw_from_file)
		else: 
			with open('data/Classical_Data_Dict_%iQBs_%iSamples' % (N_qubits, N_samples[0]), 'r') as g:
				raw_from_file = json.load(g)
				data_dict = json.loads(raw_from_file)

	elif data_type == 'Quantum_Data':
		circuit_choice = args[0][0]
	
		if (N_samples == 'infinite'):
			with open('data/Quantum_Data_Dict_%iQBs_Exact_%sCircuit' % (N_qubits, circuit_choice), 'r') as f:
				raw_from_file = json.load(f)
				data_dict = json.loads(raw_from_file)
		else: 
			with open('data/Quantum_Data_Dict_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_samples[0], circuit_choice), 'r') as g:
				raw_from_file = json.load(g)
				data_dict = json.loads(raw_from_file)
	else: raise IOError('Please enter either \'Quantum_Data\' or \'Classical_Data\' for \'data_type\' ')

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
def DataImport(data_type, N_qubits, N_data_samples, *args):
    
	data_exact_dict = DataDictFromFile(data_type, N_qubits, 'infinite', args)
	if data_type == 'Classical_Data':

		data_samples_orig = list(np.loadtxt('data/Classical_Data_%iQBs_%iSamples' % (N_qubits, N_data_samples), dtype = str))
		data_samples = SampleListToArray(data_samples_orig, N_qubits)
	
	elif data_type == 'Quantum_Data':
		circuit_choice = args[0]

		data_samples_orig = list(np.loadtxt('data/Quantum_Data_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_data_samples, circuit_choice), dtype = str))
		data_samples = SampleListToArray(data_samples_orig, N_qubits)


	else: raise IOError('Please enter either \'Quantum_Data\' or \'Classical_Data\' for \'data_type\' ')
    
	return data_samples, data_exact_dict 

## Reads kernel dictionary from file
def KernelDictFromFile(qc, N_samples, kernel_choice):
	N_qubits = len(qc.qubits())
	#reads kernel dictionary from file
	N_kernel_samples = N_samples[-1]

	if (N_kernel_samples == 'infinite'):
		with open('kernel/%sKernel_Dict_%iQBs_Exact' % (kernel_choice[0], N_qubits), 'r') as f:
			_, keys, values = FileLoad(f)
	else:
		with open('kernel/%sKernel_Dict_%iQBs_%iKernelSamples' % (kernel_choice[0], N_qubits, N_kernel_samples), 'r') as f:
			_, keys, values = FileLoad(f)

	return dict(zip(*[keys, values]))

def ConvertKernelDictToArray(N_qubits, N_kernel_samples, kernel_choice):
	'''This function converts a dictionary of kernels to a numpy array'''
	N_samples = [0, 0, 0, N_kernel_samples]
	#read kernel matrix in from file as dictionary
	kernel_dict = KernelDictFromFile(N_qubits, N_samples, kernel_choice)
	#convert dictionary to np array
	kernel_array = np.fromiter(kernel_dict.values(), dtype = float).reshape((2**N_qubits, 2**N_qubits))

	return  kernel_array

def ParamsFromFile(N_qubits):
	Params = np.load('Parameters_%iQubits.npz' % (N_qubits))
	J_i = Params['J_init']
	b_i = Params['b_init']
	g_x_i = Params['gamma_x_init']
	g_y_i = Params['gamma_y_init']
	
	return J_i, b_i, g_x_i, g_y_i


def TrainingDataFromFile(cost_func, qc, kernel_type,N_kernel_samples, N_data_samples, N_born_samples, batch_size, N_epochs):
	'''This function reads in all information generated during the training process for a specified set of parameters'''

	trial_name = "outputs/Output_%sCost_%sDevice_%skernel_%ikernel_samples_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs" \
				%(cost_func,\
				qc.name,\
				kernel_type,\
				N_kernel_samples,\
				N_born_samples,\
				N_data_samples,\
				batch_size,\
				N_epochs)

	with open('%s/info' %trial_name, 'w') as training_data_file:
		training_data = training_data_file.readlines()
		print(training_data)

	circuit_params = {}
	loss = {}
	loss[('%s' %cost_func, 'Train')] = np.loadtxt('%s/loss/%s/train' 	%(trial_name,cost_func),  dtype = float)
	loss[('%s' %cost_func, 'Test')] = np.loadtxt('%s/loss/%s/test' 	%(trial_name,cost_func),  dtype = float)
	loss[('TV')] 					= np.loadtxt('%s/loss/TV' ,  dtype = float)

	for epoch in range(0, N_epochs - 1):
		circuit_params[('J', epoch)] 		= np.loadtxt('%s/params/weights/epoch%s' 	%(trial_name, epoch), dtype = float)
		circuit_params[('b', epoch)] 		= np.loadtxt('%s/params/biases/epoch%s' 	%(trial_name, epoch), dtype = float)
		circuit_params[('gamma_x', epoch)] 	= np.loadtxt('%s/params/gammaX/epoch%s' 	%(trial_name, epoch), dtype = float)
		circuit_params[('gamma_y', epoch)] 	= np.loadtxt('%s/params/gammaY/epoch%s' 	%(trial_name, epoch), dtype = float)


	return loss, circuit_params
