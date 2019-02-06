## @package file_operations_in import functions 
#
# A collection of functions for imported pre-computed data

import numpy as np
import ast
import sys
import json
from auxiliary_functions import SampleListToArray
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
		data_samples = SampleListToArray(data_samples_orig, N_qubits, 'int')
	
	elif data_type == 'Quantum_Data':
		circuit_choice = args[0]

		data_samples_orig = list(np.loadtxt('data/Quantum_Data_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_data_samples, circuit_choice), dtype = str))
		data_samples = SampleListToArray(data_samples_orig, N_qubits, 'int')


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

def FindTrialNameFile(cost_func, N_epochs,learning_rate, qc, kernel_type, N_samples, stein_params, sinkhorn_eps):
	'''This function creates the file neame to be found with the given parameters'''

	[N_data_samples, N_born_samples, batch_size, N_kernel_samples] = N_samples
	stein_score		= stein_params[0]       
	stein_eigvecs	= stein_params[1] 
	stein_eta		= stein_params[2]    

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

	return trial_name

def TrainingDataFromFile(cost_func, N_epochs, learning_rate, qc, kernel_type, N_samples, stein_params, sinkhorn_eps):
	'''This function reads in all information generated during the training process for a specified set of parameters'''

	trial_name = FindTrialNameFile(cost_func, N_epochs,learning_rate, qc, kernel_type, N_samples, stein_params, sinkhorn_eps)

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
		circuit_params[('gamma_x', epoch)] 	= np.loadtxt('%s/params/gammaX/epoch%s' 	%(trial_name, epoch), dtype = float)
		circuit_params[('gamma_y', epoch)] 	= np.loadtxt('%s/params/gammaY/epoch%s' 	%(trial_name, epoch), dtype = float)

		with open('%s/probs/born/epoch%s' 	%(trial_name, epoch), 'r') as f:
			born_probs_dict, _, _ = FileLoad(f, 'probs_input')
			born_probs.append(born_probs_dict)
		with open('%s/probs/data/epoch%s' 	%(trial_name, epoch), 'r') as g:
			data_probs_dict, _, _ = FileLoad(g, 'probs_input')
			data_probs.append(data_probs_dict)

	return loss, circuit_params, born_probs, data_probs


# N_epochs1 = 200
# learning_rate1 =  0.1
# data_type1 = 'Classical_Data'
# N_born_samples1 = 300
# N_data_samples1 = 300
# N_kernel_samples1 = 2000
# batch_size1 = 150
# kernel_type1 ='Quantum'
# cost_func1 = 'MMD'
# qc1 = '4q-qvm'
# stein_score1 = 'Spectral_Score' 
# stein_eigvecs1 = 3                 
# stein_eta1 = 0.01      
# sinkhorn_eps1 = 0.001

# N_samples1 =     [N_data_samples1,\
# 				N_born_samples1,\
# 				batch_size1,\
# 				N_kernel_samples1]

# stein_params1 = {}
# stein_params1[0] = stein_score1       
# stein_params1[1] = stein_eigvecs1        
# stein_params1[2] = stein_eta1   
# stein_params1[3] = kernel_type1 

# loss1, circuit_params1, born_probs1, data_probs1 = TrainingDataFromFile(cost_func1, N_epochs1, learning_rate1, qc1, \
# 																			kernel_type1, N_samples1, stein_params1, sinkhorn_eps1)

# final_probs1 = born_probs1[-1]
# data_probs_final  = data_probs1[-1]

# N_epochs2 = 200
# learning_rate2 =  0.1
# data_type2 = 'Classical_Data'
# N_born_samples2 = 300
# N_data_samples2 = 300
# N_kernel_samples2 = 2000
# batch_size2 = 150
# kernel_type2 ='Quantum'
# cost_func2 = 'Sinkhorn'
# qc2 = '4q-qvm'
# stein_score2 = 'Spectral_Score' 
# stein_eigvecs2 = 3                 
# stein_eta2 = 0.01      
# sinkhorn_eps2 = 0.1

# N_samples2 =     [N_data_samples2,\
# 				N_born_samples2,\
# 				batch_size2,\
# 				N_kernel_samples2]

# stein_params2 = {}
# stein_params2[0] = stein_score2       
# stein_params2[1] = stein_eigvecs2        
# stein_params2[2] = stein_eta2   
# stein_params2[3] = kernel_type2 


# loss2, circuit_params2, born_probs2, data_probs2 = TrainingDataFromFile(cost_func2, N_epochs2, learning_rate2, qc2, \
# 																			kernel_type2, N_samples2, stein_params2, sinkhorn_eps2)

# final_probs2 = born_probs2[-1]


# N_epochs3 = 200
# learning_rate3 =  0.1
# data_type3 = 'Classical_Data'
# N_born_samples3 = 200
# N_data_samples3 = 200
# N_kernel_samples3 = 2000
# batch_size3 = 100
# kernel_type3 ='Gaussian'
# cost_func3 = 'MMD'
# qc3 = '3q-qvm'
# stein_score3 = 'Spectral_Score' 
# stein_eigvecs3 = 3                 
# stein_eta3 = 0.01      
# sinkhorn_eps3 = 0.05

# N_samples3 =     [N_data_samples3,\
# 				N_born_samples3,\
# 				batch_size3,\
# 				N_kernel_samples3]

# stein_params3 = {}
# stein_params3[0] = stein_score2       
# stein_params3[1] = stein_eigvecs2        
# stein_params3[2] = stein_eta2   
# stein_params3[3] = kernel_type2 


# loss3, circuit_params3, born_probs3, data_probs3 = TrainingDataFromFile(cost_func3, N_epochs3, learning_rate3, qc3, \
# 																			kernel_type3, N_samples3, stein_params3, sinkhorn_eps3)

# final_probs3 = born_probs3[-1]



# N_qubits = int(qc2[0])
# plot_colour = ['r', 'b', 'g']

# plt.plot(loss1[('TV')],  '%so-' %(plot_colour[0]), label ='MMD, %i Data Points,  %i Born Samples for a %s kernel.' \
# 							%(N_samples1[0], N_samples1[1], kernel_type1))
# # plt.plot(loss1[('TV')],  '%so-' %(plot_colour[0]), label ='Sinkhorn, %i Data Points,  %i Born Samples epsilon = %.3f.' \
# # 							%(N_samples1[0], N_samples1[1], sinkhorn_eps1))

# plt.plot(loss2[('TV')],  '%sx-' %(plot_colour[1]), label ='Sinkhorn, %i Data Points,  %i Born Samples for a Hamming Cost, with epsilon %.3f' \
# 							%(N_samples2[0], N_samples2[1], sinkhorn_eps2))
# # plt.plot(loss2[('TV')],  '%so-' %(plot_colour[2]), label ='MMD, %i Data Points,  %i Born Samples for a %s kernel.' \
# # 							%(N_samples2[0], N_samples2[1], kernel_type2))	
# # plt.plot(loss3[('TV')],  '%so-' %(plot_colour[2]), label ='MMD, %i Data Points,  %i Born Samples for a %s kernel.' \
# # 							%(N_samples3[0], N_samples3[1], kernel_type3))	
# # plt.plot(loss3[('TV')],  '%sx-' %(plot_colour[2]), label ='Sinkhorn, %i Data Points,  %i Born Samples for a Hamming Cost, with epsilon %.3f' \
# # 							%(N_samples3[0], N_samples3[1], sinkhorn_eps3))

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.xlabel("Epochs")
# plt.ylabel("TV")
# plt.title("TV for %i qubits" % N_qubits)

# plt.legend()

# plt.show()


# fig, axs = plt.subplots()
# # fig, axs = plt.subplots()

# axs.clear()
# x = np.arange(len(data_probs_final))
# axs.bar(x, data_probs_final.values(), width=0.2, color= plot_colour[0], align='center')
# axs.bar(x-0.2, final_probs1.values(), width=0.2, color='b', align='center')
# axs.bar(x-0.4, final_probs2.values(), width=0.2, color='g', align='center')
# # axs.bar(x-0.6, final_probs3.values(), width=0.2, color='y', align='center')
# # axs.set_title("%i Qbs, %s Kernel, %i Data Samps, %i Born Samps" \
# # 		%(N_qubits, kernel_type[0][0], N_data_samples, N_born_samples))
# axs.set_xlabel("Outcomes")
# axs.set_ylabel("Probability")
# axs.legend(('Born Probs_1','Born Probs_2', 'Data Probs'))

# # axs.legend(('Born Probs_1','Born Probs_2','Born_probs_3', 'Data Probs'))
# axs.set_xticks(range(len(data_probs_final)))
# axs.set_xticklabels(list(data_probs_final.keys()),rotation=70)

# plt.show()