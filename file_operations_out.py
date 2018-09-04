
from train_generation import TrainingData, DataSampler, ConvertToString
from classical_kernel import GaussianKernel, GaussianKernelExact
from mmd_exact import MMDKernelExact
import sys
import json
import numpy as np

Max_qubits = 3

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
	#print the required kernel out to a file
	for N_v in range(2, Max_qubits):
		print("This is qubit, ", N_v)
		bin_visible = np.zeros((2**N_v, N_v))
		for v_index in range(0,2**N_v):
			v_string = ConvertToString(v_index, N_v)
			for v in range(0, N_v):
				bin_visible[v_index][v] = float(v_string[v])

		#The number of samples, N_samples = infinite if the exact kernel is being computed
		kernel, kernel_exact, kernel_dict, kernel_exact_dict = MMDKernelExact(N_v, bin_visible,  N_kernel_samples, kernel_choice)
		KernelDictToFile(N_v, N_kernel_samples, kernel_dict, kernel_choice)
	return

print("Kernel is printing for 1000 samples")
PrintKernel(1000, 'Quantum')
PrintKernel('infinite', 'Quantum')
np.set_printoptions(threshold=np.nan)

### This function prepares data samples according to a a specified number of samples
### for all number of visible qubits up to Max_qubits, and saves them to files
def DataDictToFile(N_v, data_dict):
	#writes data dictionary to file
	with open('Data_Dict_Exact_%iNv' % N_v, 'w') as f:
		json.dump(json.dumps(data_dict, sort_keys=True),f)
	return

def PrintDataToFiles(N_data_samples, approx):
	for i in range(1, Max_qubits):
		N = i
		N_v = N
		N_h = N - N_v
		#Define training data along with all binary strings on the visible and hidden variables from train_generation
		#M_h is the number of hidden Bernoulli modes in the data
		M_h = 8
		if (approx == 'Sampler'):
			data = DataSampler(N_v, N_h, M_h, N_data_samples)
			np.savetxt('Data_%iNv_%iSamples' % (N_v, N_data_samples),data, fmt='%d')
		elif(approx == 'Exact'):
			exact_data, bin_visible, bin_hidden, data_dist_dict = TrainingData(N_v, N_h, M_h)
			#Output exact training data (not sampled)
			np.savetxt('Data_Exact_%iNv' % (N_v), np.asarray(exact_data), fmt='%.10f')
			DataDictToFile(N_v, data_dist_dict)
		else: print('approx must be either "Exact", or "Sampler"')


#PrintDataToFiles(10, 'Exact')
