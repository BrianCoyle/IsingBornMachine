TrainingData
from train_Generation import TrainingData, DataSampler
from classical_kernel import gaussian_kernel, gaussian_kernel_exact
from mmd import MMDKernelforGradExact
import ast
import sys
import json

Max_qubits = 11
def KernelDictToFile(N_v, N_kernel_samples, kernel_dict, kernel_choice):
	#writes kernel dictionary to file
	if (N_kernel_samples == 'infinite'):
		with open('%sKernel_Exact_Dict_%iNv' % (kernel_choice[0], N_v), 'w') as f:
			k = kernel_dict.keys()
			v = kernel_dict.values()
			k1 = [str(i) for i in k]
			json.dump(json.dumps(dict(zip(*[k1,v])), sort_keys=True),f)
	else:
		with open('%sKernel_Dict_%iNv_%iKernelSamples' % (kernel_choice[0], N_v, N_kernel_samples), 'w') as f:
			k = kernel_dict.keys()
			v = kernel_dict.values()
			k1 = [str(i) for i in k]
			json.dump(json.dumps(dict(zip(*[k1,v])), sort_keys=True),f)
	return

def PrintKernel():
	#print the kernel for the Gaussian and Quantum kernels out to a file
	for N_v in range(1, Max_qubits):
		print("This is qubit, ", N_v)
		bin_visible = np.zeros((2**N_v, N_v))
		for v_string in range(0,2**N_v):
			s_temp = format(v_string,'b')
			s_temp = "0" * (N_v-len(s_temp)) + s_temp
			for v in range(0, N_v):
				bin_visible[v_string][v] = float(s_temp[v])

		N_kernel_samples = 1
		#kernel_choice1 = 'Quantum'
		kernel_choice2 = 'Gaussian'

		#The number of samples, N_samples = infinite if the exact kernel is being computed
		#kernel1, kernel_exact1, kernel_dict1, kernel_exact_dict1 = mmd_grad_kernel_comp_exact(N_v, bin_visible,  N_kernel_samples, kernel_choice1)
		#dict_to_file(N_v, 'infinite', kernel_exact_dict1, kernel_choice1)
		kernel2, kernel_exact2, kernel_dict2, kernel_exact_dict2 =MMDKernelforGradExact(N_v, bin_visible,  N_kernel_samples, kernel_choice2)
		kernel_dict_to_file(N_v, 'infinite', kernel_dict2, kernel_choice2)

	return
print_kernel()

np.set_printoptions(threshold=np.nan)
### This function prepares data samples according to a a specified number of samples
### for all number of visible qubits up to Max_qubits, and saves them to files

def DataDictToFile(N_v, data_dict):
	#writes data dictionary to file
	with open('Data_Dict_Exact_%iNv' % N_v, 'w') as f:
		json.dump(json.dumps(data_dict, sort_keys=True),f)
	return
def PrintDataToFiles():
	for i in range(1, Max_qubits):
		#N_v is the number of visible units
		#N_h is the number of hidden units
		#Convention will be that qubits {0,...,N_v} will be visible,
		#qubits {N_v+1,...,N} will be hidden
		N = i
		N_v = i
		N_h = N - N_v
		#Define training data along with all binary strings on the visible and hidden variables from train_generation
		#M_h is the number of hidden Bernoulli modes in the data
		M_h = 8
		# N_data_samples_10 = 10
		# N_data_samples_100 = 100
		# N_data_samples_1000 = 1000
		# N_data_samples_10000 = 10000
		# N_data_samples_100000 = 100000
		#
		# data_10 = DataSampler(N_v, N_h, M_h, N_data_samples_10)
		# data_100 = DataSampler(N_v, N_h, M_h, N_data_samples_100)
		# data_1000 = DataSampler(N_v, N_h, M_h, N_data_samples_1000)
		# data_10000 = DataSampler(N_v, N_h, M_h, N_data_samples_10000)
		#
		# np.savetxt('Data_%iNv_%iSamples' % (N_v, N_data_samples_10),data_10, fmt='%d')
		# np.savetxt('Data_%iNv_%iSamples' % (N_v, N_data_samples_100),data_100, fmt='%d')
		# np.savetxt('Data_%iNv_%iSamples' % (N_v, N_data_samples_1000),data_1000, fmt='%d')
		# np.savetxt('Data_%iNv_%iSamples' % (N_v, N_data_samples_10000),data_10000, fmt='%d')

		exact_data, bin_visible, bin_hidden, data_dist_dict = TrainingData(N_v, N_h, M_h)
		#Output exact training data (not sampled)
		np.savetxt('Data_Exact_%iNv' % (N_v), np.asarray(exact_data), fmt='%.10f')
		data_dict_to_file(N_v, data_dist_dict)
