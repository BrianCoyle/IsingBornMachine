from train_generation import TrainingData, DataSampler
from auxiliary_functions import  ConvertToString, EmpiricalDist, SampleListToArray
from classical_kernel import GaussianKernel, GaussianKernelExact
from mmd_functions import MMDKernel
from param_init import NetworkParams
import sys
import json
import numpy as np

Max_qubits = 9

def PrintParamsToFile():

	for qubit_index in range(2, Max_qubits):
		J = np.zeros((qubit_index, qubit_index))
		b = np.zeros((qubit_index))
		gamma_x = np.zeros((qubit_index))
		gamma_y = np.zeros((qubit_index))
		#Initialize Parameters, J, b for epoch 0 at random, gamma = constant = pi/4
		J_init, b_init, gamma_x_init, gamma_y_init = NetworkParams(qubit_index, J, b, gamma_x, gamma_y)

		np.savez('Parameters_%iQubits.npz' % (qubit_index), J_init = J_init, b_init = b_init, gamma_x_init = gamma_x_init, gamma_y_init = gamma_y_init)

	return


#PrintParamsToFile()

def KernelDictToFile(N_v, N_kernel_samples, kernel_dict, kernel_choice):
	#writes kernel dictionary to file
	if (N_kernel_samples == 'infinite'):
		with open('%sKernel_Exact_Dict_%iNv' % (kernel_choice[0], N_v), 'w') as f:
			dict_keys = kernel_dict.keys()
			dict_values = kernel_dict.values()
			k1 = [str(key) for key in dict_keys]
			print(json.dump(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True, indent=0),f))
		print(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True, indent=0))

	else:
		with open('%sKernel_Dict_%iNv_%iKernelSamples' % (kernel_choice[0], N_v, N_kernel_samples), 'w') as f:
			dict_keys = kernel_dict.keys()
			dict_values = kernel_dict.values()
			k1 = [str(key) for key in dict_keys]
			print(json.dump(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True),f))
		print(json.dumps(dict(zip(*[k1, dict_values])), sort_keys=True, indent=0))

	return

def PrintKernel(N_kernel_samples, kernel_choice):
	#print the required kernel out to a file, for all binary strings
	for N_qubits in range(2, Max_qubits):
		print("This is qubit, ", N_qubits)
		bin_visible = np.zeros((2**N_qubits, N_qubits))
		for v_index in range(0,2**N_qubits):
			v_string = ConvertToString(v_index, N_qubits)
			for v in range(0, N_qubits):
				bin_visible[v_index][v] = float(v_string[v])

		#The number of samples, N_samples = infinite if the exact kernel is being computed
		kernel, kernel_exact, kernel_dict, kernel_exact_dict = MMDKernel(N_qubits, bin_visible,  N_kernel_samples, kernel_choice)
		KernelDictToFile(N_qubits, N_kernel_samples, kernel_dict, kernel_choice)
# 	return

# print("Kernel is printing for 10 samples")
# PrintKernel(10, 'Gaussian')
# print("Kernel is printing for 100 samples")
# PrintKernel(100, 'Gaussian')
# print("Kernel is printing for 200 samples")
# PrintKernel(200, 'Gaussian')
# print("Kernel is printing for 500 samples")
# PrintKernel(500, 'Gaussian')
# print("Kernel is printing for 1000 samples")
# PrintKernel(1000, 'Gaussian')
# print("Kernel is printing for 2000 samples")
# PrintKernel(2000, 'Gaussian')

# print("Exact Kernel is Printing")
# PrintKernel('infinite', 'Gaussian')
np.set_printoptions(threshold=np.nan)

### This function prepares data samples according to a a specified number of samples
### for all number of visible qubits up to Max_qubits, and saves them to files
def DataDictToFile(N_v, data_dict, N_data_samples):
	#writes data dictionary to file
	if (N_data_samples == 'infinite'):
		with open('Data_Dict_Exact_%iNv' % N_v, 'w') as f:
			json.dump(json.dumps(data_dict, sort_keys=True),f)
	else:
		with open('Data_Dict_%iSamples_%iNv' % (N_data_samples, N_v), 'w') as f:
			json.dump(json.dumps(data_dict, sort_keys=True),f)
	return

def PrintDataToFiles():
	for N_qubits in range(2, 8):
		
		N_v = N_qubits
		N_h = N_qubits - N_v
		#Define training data along with all binary strings on the visible and hidden variables from train_generation
		#M_h is the number of hidden Bernoulli modes in the data
		M_h = 8
		data_probs, bin_visible, bin_hidden, exact_data_dict = TrainingData(N_v, N_h, M_h)
	
		data_samples10_orig 	= DataSampler(N_v, N_h, M_h, 10, data_probs, exact_data_dict)
		data_samples100_orig 	= DataSampler(N_v, N_h, M_h, 100, data_probs, exact_data_dict)
		data_samples200_orig 	= DataSampler(N_v, N_h, M_h, 200, data_probs, exact_data_dict)
		data_samples500_orig 	= DataSampler(N_v, N_h, M_h, 500, data_probs, exact_data_dict)
		data_samples1000_orig	= DataSampler(N_v, N_h, M_h, 1000, data_probs, exact_data_dict)
		data_samples2000_orig 	= DataSampler(N_v, N_h, M_h, 2000, data_probs, exact_data_dict)
		data_samples3000_orig	= DataSampler(N_v, N_h, M_h, 3000, data_probs, exact_data_dict)
		data_samples4000_orig 	= DataSampler(N_v, N_h, M_h, 4000, data_probs, exact_data_dict)
		data_samples5000_orig 	= DataSampler(N_v, N_h, M_h, 5000, data_probs, exact_data_dict)
		data_samples6000_orig 	= DataSampler(N_v, N_h, M_h, 6000, data_probs, exact_data_dict)	
		data_samples8000_orig 	= DataSampler(N_v, N_h, M_h, 8000, data_probs, exact_data_dict)
		data_samples10000_orig 	= DataSampler(N_v, N_h, M_h, 10000, data_probs, exact_data_dict)		
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 10),data_samples10_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 100),data_samples100_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 200),data_samples100_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 500),data_samples100_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 1000),data_samples1000_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 2000),data_samples2000_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 3000),data_samples3000_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 4000),data_samples4000_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 5000),data_samples5000_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 6000),data_samples6000_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 8000),data_samples8000_orig, fmt='%s')
		np.savetxt('Data_%iNv_%iSamples' % (N_v, 10000),data_samples10000_orig, fmt='%s')

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
		
		emp_dist10 = EmpiricalDist(data_samples10, N_v)
		emp_dist100 = EmpiricalDist(data_samples100, N_v)
		emp_dist200 = EmpiricalDist(data_samples200, N_v)
		emp_dist500 = EmpiricalDist(data_samples500, N_v)

		emp_dist1000 = EmpiricalDist(data_samples1000, N_v)
		emp_dist2000 = EmpiricalDist(data_samples2000, N_v)
		emp_dist3000 = EmpiricalDist(data_samples3000, N_v)
		emp_dist4000 = EmpiricalDist(data_samples4000, N_v)
		emp_dist5000 = EmpiricalDist(data_samples5000, N_v)
		emp_dist6000 = EmpiricalDist(data_samples6000, N_v)
		emp_dist8000 = EmpiricalDist(data_samples8000, N_v)
		emp_dist10000 = EmpiricalDist(data_samples10000, N_v)

		DataDictToFile(N_v, emp_dist10, 10)
		DataDictToFile(N_v, emp_dist100, 100)
		DataDictToFile(N_v, emp_dist200, 200)
		DataDictToFile(N_v, emp_dist500, 500)
		DataDictToFile(N_v, emp_dist1000, 1000)
		DataDictToFile(N_v, emp_dist2000, 2000)
		DataDictToFile(N_v, emp_dist3000, 3000)
		DataDictToFile(N_v, emp_dist4000, 4000)
		DataDictToFile(N_v, emp_dist5000, 5000)
		DataDictToFile(N_v, emp_dist6000, 6000)
		DataDictToFile(N_v, emp_dist8000, 8000)
		DataDictToFile(N_v, emp_dist10000, 10000)

		#Output exact training data (not sampled)
		np.savetxt('Data_Exact_%iNv' % (N_v), np.asarray(data_probs), fmt='%.10f')
		DataDictToFile(N_v, exact_data_dict, 'infinite')
	return

PrintDataToFiles()


def PrintFinalParamsToFile(J, b, L, N_v, kernel_type, N_born_samples, N_epochs, N_data_samples, learning_rate):
    print("THIS IS THE DATA FOR MMD WITH %i VISIBLE QUBITS, WITH %s KERNEL, %i SAMPLES FROM THE BORN MACHINE,\
                %i DATA SAMPLES, %i NUMBER OF EPOCHS AND LEARNING RATE = %.3f" \
                        %(N_v, kernel_type[0], N_born_samples, N_epochs, N_data_samples, learning_rate))
    for epoch in range(0, N_epochs-1):
        print('The weights for Epoch', epoch ,'are :', J[:,:,epoch], '\n')
        print('The biases for Epoch', epoch ,'are :', b[:,epoch], '\n')
        print('MMD Loss for Epoch', epoch ,'is:', L[epoch], '\n')

    return
