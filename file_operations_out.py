from train_generation import TrainingData, DataSampler
from auxiliary_functions import  ConvertToString, EmpiricalDist, SampleListToArray
from cost_function_train import KernelExact
from param_init import NetworkParams
import json
import numpy as np

from pyquil.api import get_qc

Max_qubits = 9

def PrintParamsToFile():

	for qubit_index in range(2, Max_qubits):
		
		J_init, b_init, gamma_x_init, gamma_y_init = NetworkParams(qubit_index)
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

def PrintKernel(N_kernel_samples, kernel_choice):
	#print the required kernel out to a file, for all binary strings
	devices = [('%iq-qvm' %N_qubits , True) for N_qubits in range(2, 7)]
	print(devices)
	for device_params in devices:
		device_name = device_params[0]
		as_qvm_value = device_params[1]

		qc = get_qc(device_name, as_qvm = as_qvm_value)
		qubits = qc.qubits()
		N_qubits =len(qubits)
		print("This is qubit, ", N_qubits)
		bin_visible = np.zeros((2**N_qubits, N_qubits))
		for v_index in range(0, 2**N_qubits):
			v_string = ConvertToString(v_index, N_qubits)
			for v in range(0, N_qubits):
				bin_visible[v_index][v] = float(v_string[v])

		#The number of samples, N_samples = infinite if the exact kernel is being computed
		kernel, kernel_exact, kernel_dict, kernel_exact_dict = KernelExact(device_params, bin_visible,  N_kernel_samples, kernel_choice)
		KernelDictToFile(N_qubits, N_kernel_samples, kernel_dict, kernel_choice)
	return

def PrintSomeKernels(kernel_type):

	print("Kernel is printing for 10 samples")
	PrintKernel(10, kernel_type)
	print("Kernel is printing for 100 samples")
	PrintKernel(100, kernel_type)
	print("Kernel is printing for 200 samples")
	PrintKernel(200, kernel_type)
	print("Kernel is printing for 500 samples")
	PrintKernel(500, kernel_type)
	print("Kernel is printing for 1000 samples")
	PrintKernel(1000, kernel_type)
	print("Kernel is printing for 2000 samples")
	PrintKernel(2000, kernel_type)
	print("Exact Kernel is Printing")
	PrintKernel('infinite', kernel_type)
	return

#Uncomment if Gaussian Kernel needed to be printed to file
#PrintSomeKernels('Gaussian')

#Uncomment if Quantum Kernel needed to be printed to file
#PrintSomeKernels('Quantum')

np.set_printoptions(threshold=np.nan)

### This function prepares data samples according to a a specified number of samples
### for all number of visible qubits up to Max_qubits, and saves them to files
def DataDictToFile(N_qubits, data_dict, N_data_samples):
	#writes data dictionary to file
	if (N_data_samples == 'infinite'):
		with open('data/Data_Dict_Exact_%iQBs' % N_qubits, 'w') as f:
			json.dump(json.dumps(data_dict, sort_keys=True),f)
	else:
		with open('data/Data_Dict_%iSamples_%iQBs' % (N_data_samples, N_qubits), 'w') as f:
			json.dump(json.dumps(data_dict, sort_keys=True),f)
	return

def PrintDataToFiles():
	for N_qubits in range(2, 6):
		
		#Define training data along with all binary strings on the visible and hidden variables from train_generation
		#M_h is the number of hidden Bernoulli modes in the data
		M_h = 8
		N_h = 0
		data_probs, bin_visible, bin_hidden, exact_data_dict = TrainingData(N_qubits, N_h, M_h)
	
		data_samples10_orig 	= DataSampler(N_qubits, N_h, M_h, 10, data_probs, exact_data_dict)
		data_samples100_orig 	= DataSampler(N_qubits, N_h, M_h, 100, data_probs, exact_data_dict)
		data_samples200_orig 	= DataSampler(N_qubits, N_h, M_h, 200, data_probs, exact_data_dict)
		data_samples500_orig 	= DataSampler(N_qubits, N_h, M_h, 500, data_probs, exact_data_dict)
		data_samples1000_orig	= DataSampler(N_qubits, N_h, M_h, 1000, data_probs, exact_data_dict)
		data_samples2000_orig 	= DataSampler(N_qubits, N_h, M_h, 2000, data_probs, exact_data_dict)
		data_samples3000_orig	= DataSampler(N_qubits, N_h, M_h, 3000, data_probs, exact_data_dict)
		data_samples4000_orig 	= DataSampler(N_qubits, N_h, M_h, 4000, data_probs, exact_data_dict)
		data_samples5000_orig 	= DataSampler(N_qubits, N_h, M_h, 5000, data_probs, exact_data_dict)
		data_samples6000_orig 	= DataSampler(N_qubits, N_h, M_h, 6000, data_probs, exact_data_dict)	
		data_samples8000_orig 	= DataSampler(N_qubits, N_h, M_h, 8000, data_probs, exact_data_dict)
		data_samples10000_orig 	= DataSampler(N_qubits, N_h, M_h, 10000, data_probs, exact_data_dict)		
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 10),data_samples10_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 100),data_samples100_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 200),data_samples200_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 500),data_samples500_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 1000),data_samples1000_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 2000),data_samples2000_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 3000),data_samples3000_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 4000),data_samples4000_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 5000),data_samples5000_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 6000),data_samples6000_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 8000),data_samples8000_orig, fmt='%s')
		np.savetxt('data/Data_%iQBs_%iSamples' % (N_qubits, 10000),data_samples10000_orig, fmt='%s')

		data_samples10 = SampleListToArray(data_samples10_orig, N_qubits)
		data_samples100 = SampleListToArray(data_samples100_orig,  N_qubits)
		data_samples200 = SampleListToArray(data_samples200_orig,  N_qubits)
		data_samples500 = SampleListToArray(data_samples500_orig,  N_qubits)
		data_samples1000 = SampleListToArray(data_samples1000_orig, N_qubits)
		data_samples2000 = SampleListToArray(data_samples2000_orig, N_qubits)
		data_samples3000 = SampleListToArray(data_samples3000_orig, N_qubits)
		data_samples4000 = SampleListToArray(data_samples4000_orig, N_qubits)
		data_samples5000 = SampleListToArray(data_samples5000_orig, N_qubits)
		data_samples6000 = SampleListToArray(data_samples6000_orig, N_qubits)
		data_samples8000 = SampleListToArray(data_samples8000_orig, N_qubits)
		data_samples10000 = SampleListToArray(data_samples10000_orig, N_qubits)
		
		emp_dist10 = EmpiricalDist(data_samples10, N_qubits)
		emp_dist100 = EmpiricalDist(data_samples100, N_qubits)
		emp_dist200 = EmpiricalDist(data_samples200, N_qubits)
		emp_dist500 = EmpiricalDist(data_samples500, N_qubits)

		emp_dist1000 = EmpiricalDist(data_samples1000, N_qubits)
		emp_dist2000 = EmpiricalDist(data_samples2000, N_qubits)
		emp_dist3000 = EmpiricalDist(data_samples3000, N_qubits)
		emp_dist4000 = EmpiricalDist(data_samples4000, N_qubits)
		emp_dist5000 = EmpiricalDist(data_samples5000, N_qubits)
		emp_dist6000 = EmpiricalDist(data_samples6000, N_qubits)
		emp_dist8000 = EmpiricalDist(data_samples8000, N_qubits)
		emp_dist10000 = EmpiricalDist(data_samples10000, N_qubits)

		DataDictToFile(N_qubits, emp_dist10, 10)
		DataDictToFile(N_qubits, emp_dist100, 100)
		DataDictToFile(N_qubits, emp_dist200, 200)
		DataDictToFile(N_qubits, emp_dist500, 500)
		DataDictToFile(N_qubits, emp_dist1000, 1000)
		DataDictToFile(N_qubits, emp_dist2000, 2000)
		DataDictToFile(N_qubits, emp_dist3000, 3000)
		DataDictToFile(N_qubits, emp_dist4000, 4000)
		DataDictToFile(N_qubits, emp_dist5000, 5000)
		DataDictToFile(N_qubits, emp_dist6000, 6000)
		DataDictToFile(N_qubits, emp_dist8000, 8000)
		DataDictToFile(N_qubits, emp_dist10000, 10000)

		#Output exact training data (not sampled)
		np.savetxt('data/Data_Exact_%iQBs' % (N_qubits), np.asarray(data_probs), fmt='%.10f')
		DataDictToFile(N_qubits, exact_data_dict, 'infinite')
	return

# PrintDataToFiles()

def PrintFinalParamsToFile(cost_func, N_epochs, loss, circuit_params, born_probs_list, empirical_probs_list, device_params, kernel_type, N_samples, learning_rate):
	[N_data_samples, N_born_samples, batch_size, N_kernel_samples] = N_samples
	print("The data is:             \n \
	cost function:	        %s      \n \
	chip:        	        %s      \n \
	kernel:      		    %s      \n \
	N kernel samples:       %i      \n \
	N Born Samples:         %i      \n \
	N Data samples:         %i      \n \
	Batch size:             %i      \n \
	Epochs:                 %i      \n \
	Learning rate:          %.3f     " \
	%(cost_func,\
	device_params[0],\
	kernel_type,\
	N_kernel_samples,\
	N_born_samples,\
	N_data_samples,\
	batch_size,\
	N_epochs,\
	learning_rate))

	for epoch in range(0, N_epochs-1):
		print('%s Loss for Epoch on Train Batch for epoch' %(cost_func), epoch ,'is:', loss[('%s'% cost_func, 'Train')][epoch], '\n')
		print('%s Loss for Epoch on Test Batch for epoch' %(cost_func), epoch ,'is:', loss[('%s'% cost_func, 'Test')][epoch], '\n')
		print('Born Exact probabilities for epoch ', epoch, 'is', born_probs_list[epoch], '\n')
		print('Born Approximate probabilities for epoch ', epoch, 'is', empirical_probs_list[epoch], '\n')
		print('The weights for Epoch', epoch ,'are :', circuit_params[('J', epoch)], '\n')
		print('The biases for Epoch', epoch ,'are :', circuit_params[('b', epoch)], '\n')
		print('The gamma_x\'s for Epoch', epoch ,'are :', circuit_params[('gamma_x', epoch)], '\n')
		print('The gamma_y\'s for Epoch', epoch ,'are :', circuit_params[('gamma_y', epoch)], '\n')



	return