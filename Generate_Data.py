
from Sample_Gen import born_sampler, plusminus_sample_gen
from Train_Generation import training_data, data_sampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import sys
import json
np.set_printoptions(threshold=np.nan)

### This function prepares data samples according to a a specified number of samples
### for all number of visible qubits up to Max_qubits, and saves them to files

console_output= sys.stdout
Max_qubits = 11

def data_dict_to_file(N_v, data_dict):
	#writes data dictionary to file
	with open('Data_Dict_Exact_%iNv' % N_v, 'w') as f:
		json.dump(json.dumps(data_dict, sort_keys=True),f)
	return

def data_dict_from_file(N_v):
	#reads data dictionary from file
	with open('Data_Dict_Exact_%iNv' % N_v, 'r') as f:
		raw_from_file = json.load(f)
		data_dict = json.loads(raw_from_file)
	return data_dict


def print_all_data_to_files():
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
		# data_10 = data_sampler(N_v, N_h, M_h, N_data_samples_10)
		# data_100 = data_sampler(N_v, N_h, M_h, N_data_samples_100)
		# data_1000 = data_sampler(N_v, N_h, M_h, N_data_samples_1000)
		# data_10000 = data_sampler(N_v, N_h, M_h, N_data_samples_10000)
		#
		# np.savetxt('Data_%iNv_%iSamples' % (N_v, N_data_samples_10),data_10, fmt='%d')
		# np.savetxt('Data_%iNv_%iSamples' % (N_v, N_data_samples_100),data_100, fmt='%d')
		# np.savetxt('Data_%iNv_%iSamples' % (N_v, N_data_samples_1000),data_1000, fmt='%d')
		# np.savetxt('Data_%iNv_%iSamples' % (N_v, N_data_samples_10000),data_10000, fmt='%d')

		exact_data, bin_visible, bin_hidden, data_dist_dict = training_data(N_v, N_h, M_h)

		#Output exact training data (not sampled)
		np.savetxt('Data_Exact_%iNv' % (N_v), np.asarray(exact_data), fmt='%.10f')
		data_dict_to_file(N_v, data_dist_dict)
		NEW_DATA = data_dict_from_file(N_v)
		print("NEW_DATA is", NEW_DATA)
#print_all_data_to_files()
