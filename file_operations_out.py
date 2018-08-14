import ast
import sys
import json

def KernelDictFromFile(N_v, N_kernel_samples, kernel_choice):
	#reads kernel dictionary from file
	if (N_kernel_samples == 'infinite'):
		with open('%sKernel_Exact_Dict_%iNv' % (kernel_choice[0], N_v), 'r') as f:
			data = json.load(f)
			kernel_dict = json.loads(data)
			k = kernel_dict.keys()
			v = kernel_dict.values()
			k1 = [eval(i) for i in k]
	else:
		with open('%sKernel_Dict_%iNv_%iKernelSamples' % (kernel_choice[0], N_v, N_kernel_samples), 'r') as f:
			data = json.load(f)
			kernel_dict = json.loads(data)
			k = kernel_dict.keys()
			v = kernel_dict.values()
			k1 = [eval(i) for i in k]
	return dict(zip(*[k1,v]))

def DataDictFromFile(N_v):
	#reads data dictionary from file
	with open('Data_Dict_Exact_%iNv' % N_v, 'r') as f:
		raw_from_file = json.load(f)
		data_dict = json.loads(raw_from_file)
	return data_dict
