from pyquil.quil import Program
import numpy as np
from pyquil.api import get_qc
from numpy.linalg import inv

from train_generation import TrainingData, DataSampler
from classical_kernel import GaussianKernel
from file_operations_in import KernelDictFromFile, DataImport

from auxiliary_functions import SampleArrayToList, StringToList,  ConvertToString


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

def ComputeDeltaTerms(N_qubits, sample_list_1, sample_list_2, kernel_dict):
    '''This kernel computes the shifted value of a kernel for each argument'''
    delta_x_kernel = {}
    delta_y_kernel = {}
    x_shifted_kernel = {}
    y_shifted_kernel = {}
    xy_shifted_kernel = {}

    x_shifted_sum = {}
    y_shifted_sum = {}
    xy_shifted_sum = {}

    kernel_shifted_trace = {}
  
    count = 0
    
    for sample_1 in sample_list_1:
        for sample_2 in sample_list_2:
            x_shifted_sum[(count, sample_1, sample_2)] = 0
            y_shifted_sum[(count, sample_1, sample_2)] = 0
            xy_shifted_sum[(count, sample_1, sample_2)] = 0
            # kernel_shifted_trace[(sample_1, sample_2)] = 0

            for i in range(0, N_qubits):
              
                #Compute kernel for each sample with the ith bit shifted for the first argument, the second argument, or both together
                x_shifted_kernel[(count, sample_1, sample_2, i)]  = kernel_dict[(ShiftString(sample_1, i), sample_2)]
                y_shifted_kernel[(count, sample_1, sample_2, i)]  = kernel_dict[(sample_1, ShiftString(sample_2, i))]
                xy_shifted_kernel[(count, sample_1, sample_2, i)] = kernel_dict[(ShiftString(sample_1, i), ShiftString(sample_2, i))]

                #Compute the delta values for both the first and second arguments of the kernel being shifted for all 
                delta_x_kernel[(count, sample_1, sample_2, i)] = kernel_dict[(sample_1, sample_2)]- x_shifted_kernel[(count, sample_1, sample_2, i)]
                delta_y_kernel[(count, sample_1, sample_2, i)] = kernel_dict[(sample_1, sample_2)]- y_shifted_kernel[(count, sample_1, sample_2, i)] 
                #Sum over all shifted kernels
                x_shifted_sum[(count, sample_1, sample_2)] = x_shifted_sum[(count, sample_1, sample_2)] + x_shifted_kernel[(count, sample_1, sample_2, i)]
                y_shifted_sum[(count, sample_1, sample_2)] = y_shifted_sum[(count, sample_1, sample_2)] + y_shifted_kernel[(count, sample_1, sample_2, i)]
                xy_shifted_sum[(count, sample_1, sample_2)] = xy_shifted_sum[(count, sample_1, sample_2)] + xy_shifted_kernel[(count, sample_1, sample_2, i)]

            #Compute the final term in the Stein weighted kernel, i.e. the trace of the matrix produced when both arguments of the kernel
            #are shifted
            if (sample_1, sample_2) not in kernel_shifted_trace:
                kernel_shifted_trace[(sample_1, sample_2)]= N_qubits*kernel_dict[(sample_1, sample_2)] - x_shifted_sum[(count, sample_1, sample_2)]\
                                                 - y_shifted_sum[(count, sample_1, sample_2)] + xy_shifted_sum[(count, sample_1, sample_2)]

            count = count+1
    return delta_x_kernel, delta_y_kernel, kernel_shifted_trace
 
def DeltaDictsToMatrix(N_qubits, delta_kernel, sample_list_1, sample_list_2):
    '''This function converts a delta dictionary (i.e. a kernel shifted in one argument by all bits)
        to the corresponding matrix'''
    N_samples_1 = len(sample_list_1)
    N_samples_2 = len(sample_list_2)
    delta_matrix  = np.asarray(list(delta_kernel.values())).reshape(N_samples_1, N_samples_2, N_qubits)
    [n, m, q] = delta_matrix.shape
    delta_matrix_slices = {}
    for delta_row in range(0, n):
        for delta_column in range(0, m):
            if (sample_list_1[delta_row], sample_list_2[delta_column]) not in delta_matrix_slices:
                delta_matrix_slices[(sample_list_1[delta_row], sample_list_2[delta_column])] = delta_matrix[delta_row][delta_column][:]

            # print(delta_matrix_slices)
    return delta_matrix_slices

def ComputeSampleKernelDict(N_qubits, kernel_dict, sample_list_1, sample_list_2):
    '''This function computes the fills a kernel dictionary according to the samples in 
    data_samples_list, with values drawn from kernel_dict, '''
    sample_kernel_dict = {}
    count = 0
    #Fill the new kernel with the values actually seen in the samples
    for x_sample in sample_list_1:         
        for y_sample in sample_list_2:
            sample_kernel_dict[(count, x_sample, y_sample)] = kernel_dict[(x_sample, y_sample)]
            count = count + 1

    return sample_kernel_dict

def ConvertKernelDictToMatrix(sample_kernel_dict, N_samples_1, N_samples_2):
    '''This function converts the sampled kernel dictionary into a matrix'''
    return np.asarray(list(sample_kernel_dict.values())).reshape( N_samples_1, N_samples_2)

def ComputeInverseTerm(sample_kernel_matrix, N_samples, chi):
    return inv(sample_kernel_matrix - chi*np.identity(N_samples))
    
def ComputeKernelShift(N_qubits, kernel_dict, data_samples_list):
    '''This kernel will not be the same as the one used in the MMD, it will only be computed
    between all samples from distribution P, with every sample from the SAME distribution P'''

    shifted_kernel_for_score = {}
    shifted_kernel_sum = {}
  
    N_data_samples = len(data_samples_list)
    #Keep a count to avoid samples which have already been seen being overwritten
    count = 0
    for x_sample in data_samples_list:

        for qubit in range(0, N_qubits):
            shifted_kernel_sum[(count, x_sample, qubit)] = 0 
            for y_sample in data_samples_list:
     
                shifted_kernel_for_score[(count, x_sample, y_sample, qubit)]  = \
                kernel_dict[x_sample, y_sample] - kernel_dict[x_sample, ShiftString(y_sample, qubit)]
                #sum over all samples for a given x sample to get an estimate of the probability
                shifted_kernel_sum[(count, x_sample, qubit)] = shifted_kernel_sum[(count, x_sample, qubit)] + shifted_kernel_for_score[(count, x_sample, y_sample, qubit)]
          
            #Normalize by total number of samples
            shifted_kernel_sum[(count, x_sample, qubit)] = shifted_kernel_sum[(count, x_sample, qubit)]/N_data_samples
     
            count = count + 1

    shifted_kernel_matrix = np.asarray(list(shifted_kernel_sum.values())).reshape(N_data_samples, N_qubits)

    return shifted_kernel_matrix


def SteinMatrixtoDict(stein_score_matrix, samples_list):
    '''This Function converts the Stein Score Matrix to a Labelled Dictionary, 
    according to the nth samples, with the mth qubit shifted'''
    [n, m] = stein_score_matrix.shape
    stein_score_dict = {}
    stein_score_dict_samples = {}
    for score_row in range(0, n):
        for score_column in range(0, m):
            stein_score_dict[(score_row,  samples_list[score_row], score_column)] = stein_score_matrix[score_row][score_column]
            stein_score_dict_samples[samples_list[score_row]]  = stein_score_matrix[score_row][:]

    return stein_score_dict, stein_score_dict_samples

def ComputeApproxScoreFunc(N_qubits, sample_kernel_matrix, kernel_dict, data_samples_list, chi):
    N_data_samples = len(data_samples_list)
    
    #Compute inverse term in Stein score approximation
    inverse = ComputeInverseTerm(sample_kernel_matrix, N_data_samples, chi)
    #Compute shifted kernel term in Stein Score Approximation
    shifted_kernel_matrix  = ComputeKernelShift(N_qubits, kernel_dict, data_samples_list)

    #Compute Approximate kernel
    stein_score_matrix_approx = N_data_samples*np.dot(inverse, shifted_kernel_matrix)

    return stein_score_matrix_approx

def ComputeExactScoreFunc(N_qubits, samples_list, data_exact_dict):
    N_data_samples = len(samples_list)
    score_dict_exact = {}
    count = 0

    for x_sample in samples_list:
        for qubit in range(0, N_qubits):
            score_dict_exact[(count, x_sample,  qubit)]  = 1 - (data_exact_dict[ShiftString(x_sample, qubit)]/data_exact_dict[x_sample])
            count = count + 1    
    
    stein_score_matrix_exact = np.asarray(list(score_dict_exact.values())).reshape(N_data_samples, N_qubits)
    return stein_score_matrix_exact

def ComputeWeightedKernel(device_params, kernel_dict, data_samples_list, data_probs,  sample_list_1, sample_list_2, score_approx, chi):
    '''This kernel computes the weighted kernel for all samples from the distribution test_samples'''

   
    device_name = device_params[0]
    as_qvm_value = device_params[1]

    qc = get_qc(device_name, as_qvm = as_qvm_value)

    qubits = qc.qubits()
    N_qubits = len(qubits)

    N_samples_1 = len(sample_list_1)
    N_samples_2 = len(sample_list_2)

    sample_kernel_dict = ComputeSampleKernelDict(N_qubits, kernel_dict,  sample_list_1, sample_list_2)
    sample_kernel_matrix = ConvertKernelDictToMatrix(sample_kernel_dict,  N_samples_1, N_samples_2)
    
    delta_x_kernel_dict, delta_y_kernel_dict, kernel_shifted_trace = ComputeDeltaTerms(N_qubits, sample_list_1, sample_list_2, kernel_dict)

    delta_x_matrix_slices = DeltaDictsToMatrix(N_qubits, delta_x_kernel_dict,   sample_list_1, sample_list_2)
    delta_y_matrix_slices = DeltaDictsToMatrix(N_qubits, delta_y_kernel_dict,   sample_list_1, sample_list_2)

    if (score_approx == 'Exact_Score'):
        stein_score_matrix_1 = ComputeExactScoreFunc(N_qubits, sample_list_1, data_probs)
        stein_score_matrix_2 = ComputeExactScoreFunc(N_qubits, sample_list_2, data_probs)
    elif (score_approx == 'Approx_Score'):
        stein_score_matrix_1 = ComputeApproxScoreFunc(N_qubits, sample_kernel_matrix, kernel_dict, sample_list_1, chi)
        stein_score_matrix_2 = ComputeApproxScoreFunc(N_qubits, sample_kernel_matrix, kernel_dict, sample_list_2, chi)

    else: raise IOError('Please enter Exact_Score or Approx_Score for score_approx')

    stein_score_dict1, stein_score_dict_samples1 = SteinMatrixtoDict(stein_score_matrix_1, sample_list_1)
    stein_score_dict2, stein_score_dict_samples2 = SteinMatrixtoDict(stein_score_matrix_2, sample_list_2)

    weighted_kernel = {}

    for sample1 in sample_list_1:
        for sample2 in sample_list_2:
        
            first_term = np.dot(np.transpose(stein_score_dict_samples1[sample1]), np.dot(kernel_dict[(sample1, sample2)], stein_score_dict_samples2[sample2]))
            second_term = - np.dot(np.transpose(stein_score_dict_samples1[sample1]), delta_y_matrix_slices[(sample1, sample2)])
            third_term = - np.dot(np.transpose(delta_x_matrix_slices[(sample1, sample2)]), stein_score_dict_samples2[sample2])
            fourth_term = kernel_shifted_trace[(sample1, sample2)]
            weighted_kernel[(sample1, sample2)] = first_term + second_term + third_term + fourth_term

    return weighted_kernel

def ComputeScoreDifference(stein_score_matrix_approx, stein_score_matrix_exact, *argsv):
    '''This function computes either the Frobenius Norm, or the Infinity norm between
        the approximate and exact score matrices'''

    if ('Frobenius' in argsv):
        Norm = np.linalg.norm((stein_score_matrix_exact - stein_score_matrix_approx), ord = None)
    elif ('Infinity' in argsv):
        Norm = np.linalg.norm(stein_score_matrix_exact - stein_score_matrix_approx, ord = np.inf)

    return Norm