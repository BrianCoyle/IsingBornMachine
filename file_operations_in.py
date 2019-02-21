## @package file_operations_in import functions 
#
# A collection of functions for imported pre-computed data

import numpy as np
import ast
import sys
import json
from auxiliary_functions import SampleListToArray
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt


def FileLoad(file, *args):
	file_info = json.load(file)
	file_dict = json.loads(file_info)
	dict_keys = file_dict.keys()
	dict_values = file_dict.values()
	if 'probs_input' in args: 
		keys = file_dict.keys()
	else:
		dict_keys = file_dict.keys()
		keys = [eval(key) for key in dict_keys]
	return file_dict, keys, dict_values

## Reads data dictionary from file
#
# @param[in] N_qubits The number of qubits
# @param[in] N_samples The number of samples
#
# @returns A dictionary containing the appropriate data

def DataDictFromFile(data_type, N_qubits, N_data_samples, *args):
	if data_type == 'Bernoulli_Data':
		if (N_data_samples == 'infinite'):
			with open('data/Bernoulli_Data_Dict_%iQBs_Exact' % (N_qubits), 'r') as f:
				raw_from_file = json.load(f)
				data_dict = json.loads(raw_from_file)
		else: 
			with open('data/Bernoulli_Data_Dict_%iQBs_%iSamples' % (N_qubits, N_data_samples[0]), 'r') as g:
				raw_from_file = json.load(g)
				data_dict = json.loads(raw_from_file)

	elif data_type == 'Quantum_Data':
		circuit_choice = args[0]
	
		if (N_data_samples == 'infinite'):
			with open('data/Quantum_Data_Dict_%iQBs_Exact_%sCircuit' % (N_qubits, circuit_choice), 'r') as f:
				raw_from_file = json.load(f)
				data_dict = json.loads(raw_from_file)
		else: 
			with open('data/Quantum_Data_Dict_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_data_samples[0], circuit_choice), 'r') as g:
				raw_from_file = json.load(g)
				data_dict = json.loads(raw_from_file)
	else: raise IOError('Please enter either \'Quantum_Data\' or \'Bernoulli_Data\' for \'data_type\' ')

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
	if data_type == 'Bernoulli_Data':

		data_samples_orig = list(np.loadtxt('data/Bernoulli_Data_%iQBs_%iSamples' % (N_qubits, N_data_samples), dtype = str))
		data_samples = SampleListToArray(data_samples_orig, N_qubits, 'int')
	
	elif data_type == 'Quantum_Data':
		circuit_choice = args[0]

		data_samples_orig = list(np.loadtxt('data/Quantum_Data_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_data_samples, circuit_choice), dtype = str))
		data_samples = SampleListToArray(data_samples_orig, N_qubits, 'int')


	else: raise IOError('Please enter either \'Quantum_Data\' or \'Bernoulli_Data\' for \'data_type\' ')
    
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

def ParamsFromFile(N_qubits, circuit_choice, device_name):
	with np.load('data/Parameters_%iQbs_%sCircuit_%sDevice.npz' % (N_qubits, circuit_choice, device_name)) as circuit_params:
		J = circuit_params['J']
		b = circuit_params['b']
		gamma = circuit_params['gamma']
		delta = circuit_params['delta']

	
	return J, b, gamma, delta

def FindTrialNameFile(cost_func, data_type, data_circuit, N_epochs,learning_rate, qc, kernel_type, N_samples, stein_params, sinkhorn_eps):
	'''This function creates the file neame to be found with the given parameters'''

	[N_data_samples, N_born_samples, batch_size, N_kernel_samples] = N_samples
	stein_score		= stein_params[0]       
	stein_eigvecs	= stein_params[1] 
	stein_eta		= stein_params[2]    

	if data_type == 'Quantum_Data':
		if cost_func == 'MMD':
			trial_name = "outputs/Output_MMD_%s_%s_%s_%skernel_%ikernel_samples_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs_%.3fLR" \
						%(qc,\
						data_type,\
						data_circuit,\
						kernel_type,\
						N_kernel_samples,\
						N_born_samples,\
						N_data_samples,\
						batch_size,\
						N_epochs,\
						learning_rate)


		elif cost_func == 'Stein':
			trial_name = "outputs/Output_Stein_%s_%s_%s_%skernel_%ikernel_samples_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs_%.3fLR_%s_%iEigvecs_%.3fEta" \
						%(qc,\
						data_type,\
						data_circuit,\
						kernel_type,\
						N_kernel_samples,\
						N_born_samples,\
						N_data_samples,\
						batch_size,\
						N_epochs,\
						learning_rate,\
						stein_score,\
						stein_eigvecs, 
						stein_eta)
		


		elif cost_func == 'Sinkhorn':
			trial_name = "outputs/Output_Sinkhorn_%s_%s_%s_HammingCost_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs_%.3fLR_%.3fEpsilon" \
						%(qc,\
						data_type,\
						data_circuit,\
						N_born_samples,\
						N_data_samples,\
						batch_size,\
						N_epochs,\
						learning_rate,\
						sinkhorn_eps)
	elif data_type == 'Bernoulli_Data':
		if cost_func == 'MMD':
			trial_name = "outputs/Output_MMD_%s_%skernel_%ikernel_samples_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs_%.3fLR" \
						%(qc,\
						kernel_type,\
						N_kernel_samples,\
						N_born_samples,\
						N_data_samples,\
						batch_size,\
						N_epochs,\
						learning_rate)


		elif cost_func == 'Stein':
			trial_name = "outputs/Output_Stein_%s_%skernel_%ikernel_samples_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs_%.3fLR_%s_%iEigvecs_%.3fEta" \
						%(qc,\
						kernel_type,\
						N_kernel_samples,\
						N_born_samples,\
						N_data_samples,\
						batch_size,\
						N_epochs,\
						learning_rate,\
						stein_score,\
						stein_eigvecs, 
						stein_eta)
		


		elif cost_func == 'Sinkhorn':
			trial_name = "outputs/Output_Sinkhorn_%s_HammingCost_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs_%.3fLR_%.3fEpsilon" \
						%(qc,\
						N_born_samples,\
						N_data_samples,\
						batch_size,\
						N_epochs,\
						learning_rate,\
						sinkhorn_eps)

	else: raise IOError('\'data_type\' must be either \'Quantum_Data\' or  \'Bernoulli_Data\'')
	return trial_name

def TrainingDataFromFile(cost_func, data_type, data_circuit, N_epochs, learning_rate, qc, kernel_type, N_samples, stein_params, sinkhorn_eps):
	'''This function reads in all information generated during the training process for a specified set of parameters'''

	trial_name = FindTrialNameFile(cost_func, data_type, data_circuit, N_epochs,learning_rate, qc, kernel_type, N_samples, stein_params, sinkhorn_eps)

	with open('%s/info' %trial_name, 'r') as training_data_file:
		training_data = training_data_file.readlines()
		print(training_data)

	circuit_params = {}
	loss = {}
	loss[('%s' %cost_func, 'Train')] = np.loadtxt('%s/loss/%s/train' 	%(trial_name,cost_func),  dtype = float)
	loss[('%s' %cost_func, 'Test')] = np.loadtxt('%s/loss/%s/test' 	%(trial_name,cost_func),  dtype = float)
	loss[('TV')] 					= np.loadtxt('%s/loss/TV' 	%(trial_name) ,  dtype = float)

	born_probs = []
	data_probs = []
	for epoch in range(0, N_epochs - 1):
		circuit_params[('J', epoch)] 		= np.loadtxt('%s/params/weights/epoch%s' 	%(trial_name, epoch), dtype = float)
		circuit_params[('b', epoch)] 		= np.loadtxt('%s/params/biases/epoch%s' 	%(trial_name, epoch), dtype = float)
		circuit_params[('gamma', epoch)] 	= np.loadtxt('%s/params/gammaX/epoch%s' 	%(trial_name, epoch), dtype = float)
		circuit_params[('delta', epoch)] 	= np.loadtxt('%s/params/gammaY/epoch%s' 	%(trial_name, epoch), dtype = float)

		with open('%s/probs/born/epoch%s' 	%(trial_name, epoch), 'r') as f:
			born_probs_dict, _, _ = FileLoad(f, 'probs_input')
			born_probs.append(born_probs_dict)
		with open('%s/probs/data/epoch%s' 	%(trial_name, epoch), 'r') as g:
			data_probs_dict, _, _ = FileLoad(g, 'probs_input')
			data_probs.append(data_probs_dict)

	return loss, circuit_params, born_probs, data_probs

def ReadFromFile(N_epochs, learning_rate, data_type, data_circuit,	
				N_born_samples, N_data_samples, N_kernel_samples,
				batch_size, kernel_type, cost_func, qc, stein_score,
				stein_eigvecs, stein_eta, sinkhorn_eps):
	if type(N_epochs) is not list:
		#If the Inputs are not a list, there is only one trial
		N_trials = 1	
	else:		
		N_trials = len(N_epochs) #Number of trials to be compared is the number of elements in each input list
	

		
	if N_trials == 1:
		N_samples 		= [N_data_samples, N_born_samples, batch_size, N_kernel_samples]
		stein_params 	= {}
		stein_params[0] = stein_score      
		stein_params[1] = stein_eigvecs      
		stein_params[2] = stein_eta  
		stein_params[3] = kernel_type 
		loss, circuit_params, born_probs, data_probs = TrainingDataFromFile(cost_func,\
																								data_type, data_circuit, N_epochs, learning_rate, \
																								qc, kernel_type, N_samples, stein_params, sinkhorn_eps)
	else:	
		[loss, circuit_params, born_probs_final, data_probs_final] = [[] for _ in range(4)]
		for trial in range(N_trials):	
			N_samples 		= [N_data_samples[trial], N_born_samples[trial], batch_size[trial], N_kernel_samples[trial]]
			stein_params 	= {}
			stein_params[0] = stein_score[trial]       
			stein_params[1] = stein_eigvecs[trial]        
			stein_params[2] = stein_eta[trial]   
			stein_params[3] = kernel_type[trial] 
			loss_per_trial, circuit_params_per_trial, born_probs_per_trial, data_probs_per_trial = TrainingDataFromFile(cost_func[trial],\
																								data_type[trial], data_circuit[trial], N_epochs[trial], learning_rate[trial], \
																									qc[trial], kernel_type[trial], N_samples, stein_params, sinkhorn_eps[trial])
			loss.append(loss_per_trial)
			circuit_params.append(circuit_params_per_trial)
			born_probs_final.append(born_probs_per_trial[-1])
			data_probs_final.append(data_probs_per_trial[-1])

	return loss, born_probs_final, data_probs_final
