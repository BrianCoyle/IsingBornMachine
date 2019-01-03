## @package auxiliary_functions some additional useful functions
#
# A collection of sever additional function useful during the running of the code.

import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from pyquil.api import get_qc

import sys

def AllBinaryStrings(N_qubits):
    #Generate Array of all binary strings of length N_qubits
    binary_strings_array = np.zeros((2**N_qubits, N_qubits))
    for integer in range(0, 2**N_qubits):
        qubit_string = ConvertToString(integer, N_qubits)
        for qubit in range(0, N_qubits):
            binary_strings_array[integer][qubit] = float(qubit_string[qubit])

    return binary_strings_array

def ConvertToString(index, N_qubits):
    if type(index) is not int:
        raise TypeError('\'index\' must be an integer')
    if type(N_qubits) is not int:
        raise TypeError('\'N_qubits\' must be an integer')

    return "0" * (N_qubits-len(format(index,'b'))) + format(index,'b')

def StringToList(string):
    '''This kernel converts a binary string to a list of bits'''
    if type(string) is not str:
        raise TypeError('\'string\' must be a str')
    string_list = []

    for element in range(len(string)):
        string_list.append(int(string[element]))
    return string_list

def ShiftString(string, shift_index):
    '''This function shifts the (shift_index)th element of a string by 1 (mod 2).
        This is the shift operator ¬ for on a binary space'''
    string_list = StringToList(string)
    shifted_string_list = []
    for i in range(len(string_list)):
        if i is shift_index:
            shifted_string_list.append(str((string_list[i]+1)%2))
        else:
            shifted_string_list.append(str(string_list[i]))

    shifted_string = ''.join(shifted_string_list)

    return  shifted_string
    
## Convert string to 1D numpy array
#
# @param[in] input_object list or 1D numpy array
#
# @param[out] input_as_string converted array/list to string
#
# return Converted String

def ToString(input_object):
    '''This converts an input to string'''
    if type(input_object) is np.ndarray:
        if input_object.ndim != 1:
            raise IOError('If \'input\' is numpy array it must be 1D')
        else:
            input_as_string = ''.join([str(bit) for bit in list(input_object)])
    elif type(input_object) is list:
        input_as_string = ''.join([str(bit) for bit in input_object])
    elif type(input_object) is str:
        input_as_string = input_object
    return input_as_string

## Convert 1D numpy array to string 
#
# @param[in] string original string
#
# @param[out] string_array converted string to array
#
# return Converted Array

def StringToArray(string):
    '''This breaks a string into a numpy array'''
    string_array = np.zeros((len(string)), dtype = int)
    for bit in range(0, len(string)):
        string_array[bit] = int(string[bit])
    return string_array
## Convert list to array
#
# @param[in] original_samples_list The original list
# @param[in] N_qubits The number of qubits
#
# @param[out] sample_array The list converted into an array
#
# return Converted list
def SampleListToArray(original_samples_list, N_qubits):
    '''This function converts a list of strings, into a numpy array, where
        each [i,j] element of the new array is the jth bit of the ith string'''
    N_data_samples = len(original_samples_list)

    sample_array = np.zeros((N_data_samples, N_qubits), dtype = int)

    for sample in range(0, N_data_samples):
        for outcome in range(0, N_qubits):
            sample_array[sample, outcome] = int(original_samples_list[sample][outcome])

    return sample_array


def SampleArrayToList(sample_array):
    '''This function converts a np.array where rows are samples
    into a list of length N_samples'''
    #if number of samples in array is just one, handle separately
    if sample_array.ndim == 1:
        sample_list = []
        sample_list.append(''.join(str(e) for e in (sample_array.tolist())))
    else:
        N_samples = sample_array.shape[0]
        sample_list = []

        for sample in range(0, N_samples):
            sample_list.append(''.join(str(int(e)) for e in (sample_array[sample][:].tolist())))

    return sample_list

def EmpiricalDist(samples, N_qubits, *arg):
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
        if type(samples[0]) is not str:
            samples_new = [] 
            for sample in samples:
                samples_new.append(str(samples[sample]))
            samples = samples_new
        N_samples = len(samples)
        string_list = samples

    counts = Counter(string_list)

    for element in counts:
        '''Convert occurances to relative frequencies of binary string'''
        counts[element] = counts[element]/(N_samples)

    for index in range(0, 2**N_qubits):
        '''If a binary string has not been seen in samples, set its value to zero'''
        if ConvertToString(index, N_qubits) not in counts:
            counts[ConvertToString(index, N_qubits)] = 0

    sorted_samples_dict = {}

    keylist = sorted(counts)
    for key in keylist:
        sorted_samples_dict[key] = counts[key]

    return sorted_samples_dict


def TotalVariationCost(dict_one, dict_two):
    '''This Function computes the variation distace between two distributions'''
    if dict_one.keys() != dict_two.keys():
        raise ValueError('Keys are not the same')
    dict_abs_diff = {}
    for variable in dict_one.keys():
        dict_abs_diff[variable] = abs(dict_one[variable] - dict_two[variable])

    variation_distance = (1/4)*sum(dict_abs_diff.values())**2

    return variation_distance

def ConvertStringToVector(string):
    '''This function converts a string to a np array'''
    string_len  = len(string)
    string_vector = np.zeros(string_len, dtype = int)
    for bit in range(string_len):
        if (string[bit] == '0' or string[bit] =='1'):
            string_vector[bit] = int(string[bit])
        else: raise IOError('Please enter a binary string')
  
    return string_vector

def L2Norm(input1, input2):
    '''This function computes the L2 norm between two binary vectors'''
    if type(input1) is str and type(input2) is str:
        l2norm = (np.linalg.norm(np.abs(ConvertStringToVector(input1) - ConvertStringToVector(input2)), 2))**2
    elif type(input1) is np.ndarray and type(input2) is np.ndarray:
        if input1.ndim != 1 or input2.ndim != 1:
            raise IOError('Input vector arrays do not consist of single samples')
        else:
            l2norm = (np.linalg.norm(np.abs(input1 - input2), 2))**2
    else: raise IOError('The inputs must be 1D numpy arrays, or strings')
    return l2norm


## This function partitions an array of samples into a training array and a test array. 
#
# The last 20% of the original set is used for testing
#
# @param[in] samples A list of samples
#
# @param[out] train_test array of lists 
#
# return Split array
def TrainTestPartition(samples):
    train_test = np.split(samples, [round(len(samples)*0.8), len(samples)], axis = 0)

    return train_test

def MiniBatchSplit(samples, batch_size):
    '''This function takes the first \'batch_size\' samples out of the full sample set '''
    if (type(samples) is not np.ndarray):
        raise TypeError('The input \'samples\' must be a numpy ndarray')
    batches = np.split(samples, [batch_size, len(samples)], axis = 0)

    return batches[0]

def FindQubits(device_params):
    '''This function finds ouptuts the used qubits for a given quantum device, 
        and the Number of qubits'''
    device_name = device_params[0]
    as_qvm_value = device_params[1]

    qc = get_qc(device_name, as_qvm = as_qvm_value)
    qubits = qc.qubits()
    N_qubits = len(qubits)

    return device_name, qubits, N_qubits