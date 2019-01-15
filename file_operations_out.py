from train_generation import TrainingData, DataSampler
from auxiliary_functions import EmpiricalDist, SampleListToArray, AllBinaryStrings, FindNumQubits
from kernel_functions import KernelAllBinaryStrings
from param_init import NetworkParams
from sample_gen import BornSampler
import json
import numpy as np
import sys
import os
from pyquil.api import get_qc

max_qubits = 9

def MakeDirectory(path):
	'''Makes an directory in the given \'path\', if it does not exist already'''
	if not os.path.exists(path):
		os.makedirs(path)
	return

def PrintParamsToFile(seed):

	for qubit_index in range(2, max_qubits):
		
		J_init, b_init, gamma_x_init, gamma_y_init = NetworkParams(qubit_index, seed)
		np.savez('data/Parameters_%iQubits.npz' % (qubit_index), J_init = J_init, b_init = b_init, gamma_x_init = gamma_x_init, gamma_y_init = gamma_y_init)
	return

#PrintParamsToFile()

def KernelDictToFile(N_qubits, N_kernel_samples, kernel_dict, kernel_choice):
	#writes kernel dictionary to file
	if (N_kernel_samples == 'infinite'):
		with open('data/%sKernel_Exact_Dict_%iQBs' % (kernel_choice[0], N_qubits), 'w') as f:
			dict_keys = kernel_dict.keys()
			dict_values = kernel_dict.values()
			k1 = [str(key) for key in dict_keys]
			print(json.dump(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True, indent=0),f))
		print(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True, indent=0))

	else:
		with open('data/%sKernel_Dict_%iQBs_%iKernelSamples' % (kernel_choice[0], N_qubits, N_kernel_samples), 'w') as f:
			dict_keys = kernel_dict.keys()
			dict_values = kernel_dict.values()
			k1 = [str(key) for key in dict_keys]
			print(json.dump(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True),f))
		print(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True, indent=0))

	return

def PrintKernel(N_kernel_samples, kernel_choice, max_qubits):
	#print the required kernel out to a file, for all binary strings
	devices = [('%iq-qvm' %N_qubits , True) for N_qubits in range(2, max_qubits)]
	print(devices)

	for device_params in devices:
		N_qubits = FindNumQubits(device_params)
		print('This is qubit:', N_qubits)
		#The number of samples, N_samples = infinite if the exact kernel is being computed
		_,_, kernel_approx_dict,_ = KernelAllBinaryStrings(device_params, N_kernel_samples, kernel_choice)

		KernelDictToFile(N_qubits, N_kernel_samples, kernel_approx_dict, kernel_choice)
	return

def PrintSomeKernels(kernel_type, max_qubits):
	N_kernel_samples_list = [10, 100, 200, 500, 1000, 2000]

	for N_kernel_samples in N_kernel_samples_list:
		print("Kernel is printing for %i samples" %N_kernel_samples)
		PrintKernel(N_kernel_samples, kernel_type, max_qubits)

	print("Exact Kernel is Printing")
	PrintKernel('infinite', kernel_type, max_qubits)
	return

#Uncomment if Gaussian Kernel needed to be printed to file
# PrintSomeKernels('Gaussian', max_qubits)

#Uncomment if Quantum Kernel needed to be printed to file
#PrintSomeKernels('Quantum', max_qubits)

np.set_printoptions(threshold=np.nan)

### This function prepares data samples according to a a specified number of samples
### for all number of visible qubits up to max_qubits, and saves them to files
def DataDictToFile(data_type, N_qubits, data_dict, N_data_samples, *args):
	#writes data dictionary to file
	if data_type == 'Classical_Data':
		if (N_data_samples == 'infinite'):
			with open('data/Classical_Data_Dict_%iQBs_Exact' % N_qubits, 'w') as f:
				json.dump(json.dumps(data_dict, sort_keys=True),f)
		else:
			with open('data/Classical_Data_Dict_%iQBs_%iSamples' % (N_qubits, N_data_samples), 'w') as f:
				json.dump(json.dumps(data_dict, sort_keys=True),f)
	elif data_type == 'Quantum_Data':
		circuit_choice = args[0]
		if (N_data_samples == 'infinite'):
			with open('data/Quantum_Data_Dict_%iQBs_Exact_%sCircuit' % (N_qubits, circuit_choice), 'w') as f:
				json.dump(json.dumps(data_dict, sort_keys=True),f)
		else:
			with open('data/Quantum_Data_Dict_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_data_samples, circuit_choice), 'w') as f:
				json.dump(json.dumps(data_dict, sort_keys=True),f)

	else: raise IOError('Please enter either \'Quantum_Data\' or \'Classical_Data\' for \'data_type\' ')

	return


def PrintDataToFiles(data_type, *args):
		data_path = './data' #Create Folder for data if it does not exist
		MakeDirectory(data_path)
		N_sample_trials = [10, 20, 30, 40, 50, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]

		if data_type == 'Classical_Data':
			for N_qubits in range(2,10):

				#Define training data along with all binary strings on the visible and hidden variables from train_generation
				#M_h is the number of hidden Bernoulli modes in the data
				M_h = 8
				N_h = 0
				data_probs, exact_data_dict = TrainingData(N_qubits, N_h, M_h)
		
				for N_samples in N_sample_trials:
					data_samples = DataSampler(N_qubits, N_h, M_h, N_samples, data_probs, exact_data_dict)
					np.savetxt('data/Classical_Data_%iQBs_%iSamples' % (N_qubits, N_samples), data_samples, fmt='%s')
					data_samples_list= SampleListToArray(data_samples, N_qubits)
					emp_data_dist = EmpiricalDist(data_samples_list, N_qubits)
					DataDictToFile(data_type, N_qubits, emp_data_dist, N_samples)
				#Output exact training data (not sampled)
				np.savetxt('data/Classical_Data_%iQBs_Exact' % (N_qubits), np.asarray(data_probs), fmt='%.10f')
				DataDictToFile(data_type, N_qubits, exact_data_dict, 'infinite')

		elif data_type == 'Quantum_Data':
			
			devices = args[0]
			circuit_choice = args[1]
			for device_params in devices:
				N_qubits = FindNumQubits(device_params)
				for N_samples in N_sample_trials:
					#Set random seed differently to that which initialises the actual Born machine to be trained
					random_seed_for_data = 13
					N_Born_Samples = [0, N_samples] #BornSampler takes a list of sample values, the [1] entry is the important one
					circuit_params = NetworkParams(device_params, random_seed_for_data) #Initialise a fixed instance of parameters to learn.
					quantum_data_samples, quantum_probs_dict, quantum_probs_dict_exact = BornSampler(device_params, N_Born_Samples, circuit_params, circuit_choice)
					np.savetxt('data/Quantum_Data_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_samples, circuit_choice), quantum_data_samples, fmt='%s')
					DataDictToFile(data_type, N_qubits, quantum_probs_dict, N_samples, circuit_choice)
				np.savetxt('data/Quantum_Data_%iQBs_Exact_%sCircuit' % (N_qubits, circuit_choice), np.asarray(quantum_data_samples), fmt='%.10f')
				DataDictToFile(data_type, N_qubits, quantum_probs_dict_exact, 'infinite', circuit_choice)
		
		else: raise IOError('Please enter either \'Quantum_Data\' or \'Classical_Data\' for \'data_type\' ')

		return

# # #Uncomment if quantum data needs to be printed to file
# device_params = [('%iq-qvm' %N_qubits , True) for N_qubits in range(2, 6)]
# circuit_choice = 'QAOA'
# PrintDataToFiles('Quantum_Data', device_params, circuit_choice)

# Uncomment if classical data needs to be printed to file
# PrintDataToFiles('Classical_Data')

def PrintCircuitParamsToFile(random_seed, circuit_choice):
	devices = [('%iq-qvm' %N_qubits , True) for N_qubits in range(2, 7)]
	for device_params in devices:
		
		device_name = device_params[0]
		as_qvm_value = device_params[1]

		qc = get_qc(device_name, as_qvm = as_qvm_value)
		qubits = qc.qubits()
		N_qubits = len(qubits)
		circuit_params = NetworkParams(device_params, random_seed)
		np.savez('data/Parameters_%iQbs_%sCircuit_%sDevice.npz' % (N_qubits, circuit_choice, device_name),\
				 J = circuit_params['J'], b = circuit_params['b'], gamma_x = circuit_params['gamma_x'], gamma_y = circuit_params['gamma_y'])

	return

#Uncomment to print circuit parameters to file, corresponding to the data, if the data is quantum
# random_seed_for_data = 13
# PrintCircuitParamsToFile(random_seed_for_data, circuit_choice)


def PrintFinalParamsToFile(cost_func, N_epochs, loss, circuit_params, born_probs_list, empirical_probs_list, device_params, kernel_type, N_samples):
	'''This function prints out all information generated during the training process for a specified set of parameters'''

	[N_data_samples, N_born_samples, batch_size, N_kernel_samples] = N_samples
	trial_name = "outputs/Output_%s_%s_%skernel_%ikernel_samples_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs" \
				%(cost_func,\
				device_params[0],\
				kernel_type,\
				N_kernel_samples,\
				N_born_samples,\
				N_data_samples,\
				batch_size,\
				N_epochs)

	with open('%s/info' %trial_name, 'w') as f:
		sys.stdout  = f
		print("The data is:           			\n \
				cost function:	        %s      \n \
				chip:        	        %s      \n \
				kernel:      		    %s      \n \
				N kernel samples:       %i      \n \
				N Born Samples:         %i      \n \
				N Data samples:         %i      \n \
				Batch size:             %i      \n \
				Epochs:                 %i      " \
		%(cost_func,\
		device_params[0],\
		kernel_type,\
		N_kernel_samples,\
		N_born_samples,\
		N_data_samples,\
		batch_size,\
		N_epochs))

	loss_path = '%s/loss/%s/' %(trial_name, cost_func)
	weight_path = '%s/params/weights/' %trial_name
	bias_path = '%s/params/biases/' %trial_name
	gammax_path = '%s/params/gammaX/'  %trial_name
	gammay_path = '%s/params/gammaY/'  %trial_name

	#create directories to store output training information
	MakeDirectory(loss_path)
	MakeDirectory(weight_path)
	MakeDirectory(bias_path)
	MakeDirectory(gammax_path)
	MakeDirectory(gammay_path)


	# with open('%s/loss' %trail_name, 'w'):
	np.savetxt('%s/loss/%s/train' 	%(trial_name,cost_func),  	loss[('%s' %cost_func, 'Train')])
	np.savetxt('%s/loss/%s/test' 	%(trial_name,cost_func), 	loss[('%s' %cost_func, 'Test')] )
	for epoch in range(0, N_epochs - 1):
		np.savetxt('%s/params/weights/epoch%s' 	%(trial_name, epoch), circuit_params[('J', epoch)]	)
		np.savetxt('%s/params/biases/epoch%s' 		%(trial_name, epoch), circuit_params[('b', epoch)])
		np.savetxt('%s/params/gammaX/epoch%s' 	%(trial_name, epoch), circuit_params[('gamma_x', epoch)])
		np.savetxt('%s/params/gammaY/epoch%s' 	%(trial_name, epoch), circuit_params[('gamma_y', epoch)])

	return