import numpy as np
import matplotlib.pyplot as plt

from collections import Counter

import sys

def ConvertToString(index, N_qubits):
	return "0" * (N_qubits-len(format(index,'b'))) + format(index,'b')

def StringToList(string):
    '''This kernel converts a binary string to a list of integers'''
    string_list = []

    for element in range(len(string)):
        string_list.append(int(string[element]))
    return string_list

def SampleListToArray(original_samples_list, N_qubits):
    '''This function converts a list of strings, into a numpy array, where
        each [i,j] element of the new array is the jth bit of the ith string'''
    N_data_samples = len(original_samples_list)

    sample_array = np.zeros((N_data_samples, N_qubits), dtype = int)

    for sample in range(0, N_data_samples):
        temp_string = original_samples_list[sample]
        for outcome in range(0, N_qubits):
            sample_array[sample, outcome] = int(temp_string[outcome])

    return sample_array


def SampleArrayToList(sample_array):
	'''This function converts a np.array where rows are samples
		into a list of length N_samples'''
	N_samples = sample_array.shape[0]
	sample_list = []

	for sample in range(0, N_samples):
		sample_list.append(''.join(str(e) for e in (sample_array[sample, :].tolist())))

	return sample_list
    
def EmpiricalDist(samples, N_v, *arg):
	'''This method outputs the empirical probability distribution given samples in a numpy array
		as a dictionary, with keys as outcomes, and values as probabilities'''

	if type(samples) is not np.ndarray and type(samples) is not list:
		raise TypeError('The samples must be either a numpy array, or list')

	if type(samples) is np.ndarray:
		N_samples = samples.shape[0]
		string_list = []
		for sample in range(0, N_samples):
			'''Convert numpy array of samples, to a list of strings of the samples to put in dict'''
			string_list.append(''.join(map(str, samples[sample, :].tolist())))

	elif type(samples) is list:
		N_samples = len(samples)
		string_list = samples

	counts = Counter(string_list)

	for element in counts:
		'''Convert occurances to relative frequencies of binary string'''
		counts[element] = counts[element]/(N_samples)

	for index in range(0, 2**N_v):
		'''If a binary string has not been seen in samples, set its value to zero'''
		if ConvertToString(index, N_v) not in counts:
			counts[ConvertToString(index, N_v)] = 0

	sorted_samples_dict = {}

	keylist = sorted(counts)
	for key in keylist:
		sorted_samples_dict[key] = counts[key]

	return sorted_samples_dict
