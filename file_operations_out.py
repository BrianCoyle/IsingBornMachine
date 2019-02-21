from train_generation import TrainingData, DataSampler
from auxiliary_functions import EmpiricalDist, AllBinaryStrings, num_bytes_needed, SampleListToArray
from kernel_functions import KernelAllBinaryStrings
from param_init import NetworkParams
from sample_gen import BornSampler
import json
import numpy as np
import sys
import os
from pyquil.api import get_qc

max_qubits = 8

def MakeDirectory(path):
	'''Makes an directory in the given \'path\', if it does not exist already'''
	if not os.path.exists(path):
		os.makedirs(path)
	return

def PrintParamsToFile(seed, max_qubits):

    for qubit_index in range(2, max_qubits):

        J_init, b_init, gamma_init, delta_init = NetworkParams(qubit_index, seed)
        np.savez('data/Parameters_%iQubits.npz' % (qubit_index), J_init = J_init, b_init = b_init, gamma_init = gamma_init, delta_init = delta_init)

        return

#PrintParamsToFile(seed, max_qubits)

def KernelDictToFile(N_qubits, N_kernel_samples, kernel_dict, kernel_choice):
	#writes kernel dictionary to file
	if (N_kernel_samples == 'infinite'):
		with open('kernel/%sKernel_Dict_%iQBs_Exact' % (kernel_choice[0], N_qubits), 'w') as f:
			dict_keys = kernel_dict.keys()
			dict_values = kernel_dict.values()
			k1 = [str(key) for key in dict_keys]
			print(json.dump(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True, indent=0),f))
		print(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True, indent=0))

	else:
		with open('kernel/%sKernel_Dict_%iQBs_%iKernelSamples' % (kernel_choice[0], N_qubits, N_kernel_samples), 'w') as f:
			dict_keys = kernel_dict.keys()
			dict_values = kernel_dict.values()
			k1 = [str(key) for key in dict_keys]
			print(json.dump(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True),f))
		print(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True, indent=0))

	return

def PrintKernel(N_kernel_samples, kernel_choice, max_qubits):
	#print the required kernel out to a file, for all binary strings
	devices = [get_qc('%iq-qvm' %N_qubits , as_qvm = True) for N_qubits in range(2, max_qubits)]

	for qc in devices:
		N_qubits = len(qc.qubits())
		print('This is qubit:', N_qubits)
		#The number of samples, N_samples = infinite if the exact kernel is being computed
		_,_, kernel_approx_dict,_ = KernelAllBinaryStrings(qc, N_kernel_samples, kernel_choice)

		KernelDictToFile(N_qubits, N_kernel_samples, kernel_approx_dict, kernel_choice)
	return

def PrintSomeKernels(kernel_type, max_qubits):
	kernel_path = './kernel' #Create Folder for data if it does not exist
	MakeDirectory(kernel_path)
	N_kernel_samples_list = [10, 100, 200, 500, 1000, 2000]
	# N_kernel_samples_list = [10]

	for N_kernel_samples in N_kernel_samples_list:
		print("Kernel is printing for %i samples" %N_kernel_samples)
		PrintKernel(N_kernel_samples, kernel_type, max_qubits)

	print("Exact Kernel is Printing")
	PrintKernel('infinite', kernel_type, max_qubits)
	return

#Uncomment if Gaussian Kernel needed to be printed to file
# PrintSomeKernels('Gaussian', max_qubits)

#Uncomment if Quantum Kernel needed to be printed to file
# PrintSomeKernels('Quantum', max_qubits)


def DataDictToFile(data_type, N_qubits, data_dict, N_data_samples, *args):
	''' This function prepares data samples according to a a specified number of samples
		for all number of visible qubits up to max_qubits, and saves them to files'''

	if data_type.lower() == 'bernoulli_data':
		if (N_data_samples == 'infinite'):
			with open('data/Bernoulli_Data_Dict_%iQBs_Exact' % N_qubits, 'w') as f:
				json.dump(json.dumps(data_dict, sort_keys=True),f)
		else:
			with open('data/Bernoulli_Data_Dict_%iQBs_%iSamples' % (N_qubits, N_data_samples), 'w') as f:
				json.dump(json.dumps(data_dict, sort_keys=True),f)
	elif data_type.lower() == 'quantum_data':
		circuit_choice = args[0]
		if (N_data_samples == 'infinite'):
			with open('data/Quantum_Data_Dict_%iQBs_Exact_%sCircuit' % (N_qubits, circuit_choice), 'w') as f:
				json.dump(json.dumps(data_dict, sort_keys=True),f)
		else:
			with open('data/Quantum_Data_Dict_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_data_samples, circuit_choice), 'w') as f:
				json.dump(json.dumps(data_dict, sort_keys=True),f)

	else: raise ValueError('Please enter either \'Quantum_Data\' or \'Bernoulli_Data\' for \'data_type\' ')

	return


def PrintCircuitParamsToFile(random_seed, circuit_choice):
	quantum_computers = [get_qc('%iq-qvm' %N_qubits , as_qvm = True) for N_qubits in range(2, 7)]
	for qc in quantum_computers:
		device_name = qc.name
		qubits = qc.qubits()
		N_qubits = len(qubits)
		circuit_params = NetworkParams(qc, random_seed)
		np.savez('data/Parameters_%iQbs_%sCircuit_%sDevice.npz' % (N_qubits, circuit_choice, device_name),\
				 J = circuit_params['J'], b = circuit_params['b'], gamma = circuit_params['gamma'], delta = circuit_params['delta'])

def string_to_int_byte(string, N_qubits, byte):

    total = 0
    for qubit in range(8 * byte, min(8 * (byte + 1), N_qubits)):
        total <<= 1
        total += int(string[qubit])

    return total
    
def PrintDataToFiles(data_type, N_samples, qc, circuit_choice, N_qubits):

	binary_data_path = 'binary_data/'
	MakeDirectory(binary_data_path)
	data_path = 'data/'
	MakeDirectory(data_path)
	if data_type == 'Bernoulli_Data':
        
		#Define training data along with all binary strings on the visible and hidden variables from train_generation
		#M_h is the number of hidden Bernoulli modes in the data
		M_h = 8
		N_h = 0
		data_probs, exact_data_dict = TrainingData(N_qubits, N_h, M_h)

		data_samples = DataSampler(N_qubits, N_h, M_h, N_samples, data_probs)

		#Save data as binary files
		with open('binary_data/Bernoulli_Data_%iQBs_%iSamples' % (N_qubits, N_samples), 'wb') as f:

			for string in data_samples:
				for byte in range(num_bytes_needed(N_qubits)):

					total = string_to_int_byte(string, N_qubits, byte)

					f.write(bytes([total]))

		np.savetxt('data/Bernoulli_Data_%iQBs_%iSamples' % (N_qubits, N_samples), data_samples, fmt='%s')
		data_samples_list = SampleListToArray(data_samples, N_qubits, 'int')
		emp_data_dist = EmpiricalDist(data_samples_list, N_qubits)
		DataDictToFile(data_type, N_qubits, emp_data_dist, N_samples)

		np.savetxt('data/Bernoulli_Data_%iQBs_Exact' % (N_qubits), np.asarray(data_probs), fmt='%.10f')
		DataDictToFile(data_type, N_qubits, exact_data_dict, 'infinite')

	elif data_type.lower() == 'quantum_data':
		          
		#Set random seed differently to that which initialises the actual Born machine to be trained
		random_seed_for_data = 13
		N_Born_Samples = [0, N_samples] #BornSampler takes a list of sample values, the [1] entry is the important one
		circuit_params = NetworkParams(qc, random_seed_for_data) #Initialise a fixed instance of parameters to learn.
		quantum_data_samples, quantum_probs_dict, quantum_probs_dict_exact = BornSampler(qc, N_Born_Samples, circuit_params, circuit_choice)
		print(quantum_data_samples)

		np.savetxt('data/Quantum_Data_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_samples, circuit_choice), quantum_data_samples, fmt='%s')
		DataDictToFile(data_type, N_qubits, quantum_probs_dict, N_samples, circuit_choice)
		np.savetxt('data/Quantum_Data_%iQBs_Exact_%sCircuit' % (N_qubits, circuit_choice), np.asarray(quantum_data_samples), fmt='%.10f')
		DataDictToFile(data_type, N_qubits, quantum_probs_dict_exact, 'infinite', circuit_choice)

	else: raise ValueError('Please enter either \'Quantum_Data\' or \'Bernoulli_Data\' for \'data_type\' ')

	return

def PrintAllDataToFiles(data_type, max_qubits, *args):
	'''
	This function prints all data samples to files, for either Quantum or Classical Data
	for all number of qubits between 2 and max_qubits.
	'''
	N_sample_trials = [10, 20, 30, 40, 50, 80, 100, 200, 300, 400, 500, 600, 700, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]

	for N_qubits in range(2, max_qubits):
		for N_samples in N_sample_trials:
			if data_type == 'Quantum_Data':
				qc = get_qc('%iq-qvm' %N_qubits , as_qvm = True)
				circuit_choice = args[0]
			
				print('Quantum Data is printing for %i qubits on qc %s using circuit choice %s' %(N_qubits, qc.name, circuit_choice))
				PrintDataToFiles('Quantum_Data', N_samples, qc, circuit_choice, N_qubits)

			elif data_type == 'Bernoulli_Data':
				qc 				= None
				circuit_choice 	= None
				print('Bernoulli Data is printing for %i qubits' %N_qubits)
				PrintDataToFiles('Bernoulli_Data', N_samples, qc, circuit_choice, N_qubits)

# #Uncomment if quantum data needs to be printed to file
# PrintAllDataToFiles('Quantum_Data', max_qubits, 'IQP')

# Uncomment if classical data needs to be printed to file
# PrintAllDataToFiles('Bernoulli_Data', max_qubits)


# #Uncomment to print circuit parameters to file, corresponding to the data, if the data is quantum
# random_seed_for_data = 13
# PrintCircuitParamsToFile(random_seed_for_data, 'IQP')

def MakeTrialNameFile(cost_func,data_type, data_circuit, N_epochs,learning_rate, qc, kernel_type, N_samples, stein_params, sinkhorn_eps):
	'''This function prints out all information generated during the training process for a specified set of parameters'''

	[N_data_samples, N_born_samples, batch_size, N_kernel_samples] = N_samples
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

			path_to_output = './%s/' %trial_name
			MakeDirectory(path_to_output)

			with open('%s/info' %trial_name, 'w') as f:
				sys.stdout  = f
				print("The data is:cost function:MMD chip:%s Data_type: %s Data Circuit: %s kernel:%s N kernel samples:%i N Born Samples:%i N Data samples:%i\
						Batch size:%i Epochs:%i Adam Learning Rate:%.3f" 
				%(qc,\
				data_type,\
				data_circuit,\
				kernel_type,\
				N_kernel_samples,\
				N_born_samples,\
				N_data_samples,\
				batch_size,\
				N_epochs,\
				learning_rate))
						
		elif cost_func == 'Stein':
			stein_score		= stein_params[0]       
			stein_eigvecs	= stein_params[1] 
			stein_eta		= stein_params[2]    
			
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
			path_to_output = './%s/' %trial_name
			MakeDirectory(path_to_output)

			with open('%s/info' %trial_name, 'w') as f:
				sys.stdout  = f
				print("The data is: cost function: Stein, chip:%s Data_type: %s Data Circuit: %s kernel:%s N kernel samples:%i \n N Born Samples:%i N Data samples:%i\
				Batch size:%iEpochs:%iAdam Learning Rate:%.3fStein Score:%sN Nystrom Eigvecs:%iStein Eta:%.3f" 

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
				stein_eigvecs, \
				stein_eta))

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

			path_to_output = './%s/' %trial_name
			MakeDirectory(path_to_output)

			with open('%s/info' %trial_name, 'w') as f:
				sys.stdout  = f
				print("The data is: cost function:Sinkhorn Data_type: %s Data Circuit: %s chip: %s kernel: %s N kernel samples: %i \
				N Born Samples: %i N Data samples: %i Batch size: %i Epochs: %i Adam Learning Rate: %.3f Sinkhorn Epsilon: %.3f" 

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
				sinkhorn_eps))
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

			path_to_output = './%s/' %trial_name
			MakeDirectory(path_to_output)

			with open('%s/info' %trial_name, 'w') as f:
				sys.stdout  = f
				print("The data is:cost function:MMD chip:%s kernel:%s N kernel samples:%i N Born Samples:%i N Data samples:%i\
						Batch size:%i Epochs:%i Adam Learning Rate:%.3f" 
				%(qc,\
				kernel_type,\
				N_kernel_samples,\
				N_born_samples,\
				N_data_samples,\
				batch_size,\
				N_epochs,\
				learning_rate))
						
		elif cost_func == 'Stein':
			stein_score		= stein_params[0]       
			stein_eigvecs	= stein_params[1] 
			stein_eta		= stein_params[2]    
			
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
			path_to_output = './%s/' %trial_name
			MakeDirectory(path_to_output)

			with open('%s/info' %trial_name, 'w') as f:
				sys.stdout  = f
				print("The data is: cost function: Stein, chip:%s  kernel:%s N kernel samples:%i \n N Born Samples:%i N Data samples:%i\
				Batch size:%iEpochs:%iAdam Learning Rate:%.3fStein Score:%sN Nystrom Eigvecs:%iStein Eta:%.3f" 

				%(qc,\
				kernel_type,\
				N_kernel_samples,\
				N_born_samples,\
				N_data_samples,\
				batch_size,\
				N_epochs,\
				learning_rate,\
				stein_score,\
				stein_eigvecs, \
				stein_eta))

		elif cost_func == 'Sinkhorn':
			trial_name = "outputs/Output_Sinkhorn_%s_HammingCost_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs_%.3fLR_%.3fEpsilon" \
						%(qc,\
						N_born_samples,\
						N_data_samples,\
						batch_size,\
						N_epochs,\
						learning_rate,\
						sinkhorn_eps)

			path_to_output = './%s/' %trial_name
			MakeDirectory(path_to_output)

			with open('%s/info' %trial_name, 'w') as f:
				sys.stdout  = f
				print("The data is: cost function: Sinkhorn chip: %s kernel: %sN kernel samples: %i N Born Samples: %iN Data samples: %i Batch size: %i  \n \
				Epochs: %i Adam Learning Rate:  %.3f Sinkhorn Epsilon:  %.3f" 

				%(qc,\
				kernel_type,\
				N_kernel_samples,\
				N_born_samples,\
				N_data_samples,\
				batch_size,\
				N_epochs,\
				learning_rate,\
				sinkhorn_eps))


	else: raise ValueError('\'data_type\' must be either \'Quantum_Data\' or  \'Bernoulli_Data\'')
	return trial_name

def PrintFinalParamsToFile(cost_func, data_type, data_circuit, N_epochs, learning_rate, loss, circuit_params, data_exact_dict, born_probs_list, empirical_probs_list, qc, kernel_type, N_samples, stein_params, sinkhorn_eps):
	'''This function prints out all information generated during the training process for a specified set of parameters'''

	  
	trial_name = MakeTrialNameFile(cost_func, data_type, data_circuit, N_epochs,learning_rate, qc.name, kernel_type, N_samples, stein_params, sinkhorn_eps)

	loss_path = '%s/loss/%s/' %(trial_name, cost_func)
	weight_path = '%s/params/weights/' %trial_name
	bias_path = '%s/params/biases/' %trial_name
	gammax_path = '%s/params/gammaX/'  %trial_name
	gammay_path = '%s/params/gammaY/'  %trial_name

	born_probs_path = '%s/probs/born/'  %trial_name
	data_probs_path = '%s/probs/data/'  %trial_name

	#create directories to store output training information
	MakeDirectory(loss_path)
	# MakeDirectory(tv_path)

	MakeDirectory(weight_path)
	MakeDirectory(bias_path)
	MakeDirectory(gammax_path)
	MakeDirectory(gammay_path)

	MakeDirectory(born_probs_path)
	MakeDirectory(data_probs_path)


	# with open('%s/loss' %trail_name, 'w'):
	np.savetxt('%s/loss/%s/train' 	%(trial_name,cost_func),  	loss[('%s' %cost_func, 'Train')])
	np.savetxt('%s/loss/%s/test' 	%(trial_name,cost_func), 	loss[('%s' %cost_func, 'Test')] )
	
	np.savetxt('%s/loss/TV' %(trial_name),  loss[('TV')]) #Print Total Variation of Distributions during training

	data_path = '%s/data' %(trial_name)
	for epoch in range(0, N_epochs - 1):
		np.savetxt('%s/params/weights/epoch%s' 	%(trial_name, epoch), circuit_params[('J', epoch)])
		np.savetxt('%s/params/biases/epoch%s' 	%(trial_name, epoch), circuit_params[('b', epoch)])
		np.savetxt('%s/params/gammaX/epoch%s' 	%(trial_name, epoch), circuit_params[('gamma', epoch)])
		np.savetxt('%s/params/gammaY/epoch%s' 	%(trial_name, epoch), circuit_params[('delta', epoch)])

		with open('%s/probs/born/epoch%s' 	%(trial_name, epoch), 'w') as f:
				json.dump(json.dumps(empirical_probs_list[epoch], sort_keys=True),f)
		with open('%s/probs/data/epoch%s' 	%(trial_name, epoch), 'w') as f:				
				json.dump(json.dumps(data_exact_dict, sort_keys=True),f)
	return
