from pyquil.quil import Program
import numpy as np
from pyquil.api import get_qc
from numpy.linalg import inv

from train_generation import DataSampler
from classical_kernel import GaussianKernelArray, GaussianKernel
from file_operations_in import KernelDictFromFile, DataImport

from auxiliary_functions import ShiftString, SampleArrayToList, StringToList, ConvertToString, ToString, EmpiricalDist
from spectral_stein_score import SpectralSteinScore

import matplotlib.pyplot as plt

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
                delta_matrix_slices[(sample_list_1[delta_row], sample_list_2[delta_column])] =\
                     delta_matrix[delta_row][delta_column][:]

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
    return np.asarray(list(sample_kernel_dict.values())).reshape(N_samples_1, N_samples_2)

def ComputeInverseTerm(kernel_array, N_samples, chi):
    '''This function computes the inverse matrix required by the Stein Score Approximator'''
    return inv(kernel_array - chi*np.identity(N_samples))
    
def ComputeKernelShift(samples, kernel_choice, stein_sigma):
    '''This kernel will not be the same as the one used in the MMD, it will only be computed
    between all samples from distribution P, with every sample from the SAME distribution P'''
    N_samples = len(samples)
    N_qubits = len(samples[0])
    shifted_kernel_for_score = {}

    shifted_kernel_for_score = np.zeros((N_qubits, N_samples, N_samples))

    # shifted_kernel_sum = {}
  
    for sample_1_index in range(0, N_samples):
            # shifted_kernel_sum[(count, sample, qubit)] = 0 
        for sample_2_index in range(0, N_samples):
            for qubit in range(0, N_qubits):

                sample_1 = ToString(samples[sample_1_index])
                
                sample_2 = ToString(samples[sample_2_index])

                # if (sample_1, ShiftString(sample_2, qubit)) in kernel_dict.keys():
                #     shifted_kernel_for_score[qubit][sample_1_index][sample_2_index]  = \
                #     kernel_dict[sample_1, sample_2] - kernel_dict[sample_1, ShiftString(sample_2, qubit)]

                # else:
                shifted_kernel_for_score[qubit][sample_1_index][sample_2_index]  = \
                    GaussianKernel(sample_1, sample_2, stein_sigma) -\
                    GaussianKernel(sample_1, ShiftString(sample_2, qubit), stein_sigma)
                #sum over all samples for a given x sample to get an estimate of the probability
                # shifted_kernel_sum[(count, sample, qubit)] = shifted_kernel_sum[(count, sample, qubit)]\
                #         + shifted_kernel_for_score[(count, x_sample, y_sample, qubit)]
          
        
            #Normalize by total number of samples
            # shifted_kernel_sum[(count, sample, qubit)] = shifted_kernel_sum[(count, sample, qubit)]/N_samples
    shifted_kernel_array = shifted_kernel_for_score.sum(axis = 2)/N_samples

    # shifted_kernel_matrix = np.asarray(list(shifted_kernel_sum.values())).reshape(N_samples, N_qubits)

    return shifted_kernel_array

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

def IdentitySteinScore(samples, kernel_type, chi, stein_sigma):
    '''This function computes the Stein Score matrix for all samples, based
    on the method of inverting Stein's identity'''

    N_samples = len(samples)

    #compute kernel matrix between all samples
    kernel_array = GaussianKernelArray(samples, samples, stein_sigma)

    #Compute inverse term in Stein score approximation
    inverse = ComputeInverseTerm(kernel_array, N_samples, chi)
    #Compute shifted kernel term in Stein Score Approximation
    shifted_kernel_matrix  = ComputeKernelShift(samples, kernel_type, stein_sigma)

    #Compute Approximate kernel
    stein_score_array_identity = N_samples*np.dot(inverse, np.transpose(shifted_kernel_matrix))

    return stein_score_array_identity


def MassSteinScoreSingleSample(sample, data_dict):
    '''This computes the exact Stein Score function in the discrete case for a single 
    sample which is a 1D numpy array, based on probability *mass* function data_dict'''
    if type(sample) is np.ndarray and sample.ndim != 1:
        raise TypeError('If \'sample\' is a numpy array, it must be 1 - Dimensional')
    N_qubits = len(sample)
    sample_string = ToString(sample)
    stein_score_sample_mass = np.zeros((N_qubits))
    for bit_index in range(0, N_qubits):
        shifted_string = ShiftString(sample_string, bit_index)
        stein_score_sample_mass[bit_index] = 1 - data_dict[shifted_string]/data_dict[sample_string]

    return stein_score_sample_mass

def MassSteinScore(samples, data_dict):
    '''This computes the Stein Matrix for all samples, based on probability *mass* function '''
    N_samples = len(samples)
    N_qubits  = len(samples[0])
    stein_score_mass_array = np.zeros((N_samples, N_qubits))
    if type(samples) is not np.ndarray and type(samples) is not list:
        raise TypeError('\'samples\' must be a numpy array or a list')

    for sample_index in range(0, N_samples):
        stein_score_mass_array[sample_index][:] =\
                MassSteinScoreSingleSample(samples[sample_index][:], data_dict)
    return stein_score_mass_array


def ComputeScoreDifference(array_1, array_2, norm_type):
    '''This function computes either the Frobenius Norm, Infinity norm or a simple sum difference
        between the two arrays'''

    if (norm_type is 'Frobenius'):
        Norm = np.linalg.norm((array_1 - array_2), ord = None)
    elif (norm_type is 'Infinity'):
        Norm = np.linalg.norm(array_1 - array_2, ord = np.inf)
    else: raise IOError('\'norm_type\' must be \'Frobenius\', \'Infinity\'')
    return Norm

def CheckScoreApproximationDifference(max_qubits, eta):
    N_qubits_list = [i for i in range(2, max_qubits)]
    stein_sigma = [0.1, 10, 100]
    N_kernel_samples = 100
    N_data_samples = [10, 20]
    
    # N_data_samples = [10, 100, 200, 300, 400]
    kernel_type = 'Gaussian'
    data_type = 'Classical_Data'
   
    spectral_exact_diff = np.zeros((len(N_qubits_list), len(N_data_samples)))
    identity_exact_diff = np.zeros((len(N_qubits_list), len(N_data_samples)))
    mass_exact_diff = np.zeros((len(N_qubits_list), len(N_data_samples)))

    for qubit_index in range(0, len(N_qubits_list)):
        N_qubits = N_qubits_list[qubit_index]
        for sample_index in range(0, len(N_data_samples)):
            J = N_qubits + 2
            N_samples = N_data_samples[sample_index]
            data_samples, data_dict = DataImport(data_type, N_qubits, N_samples)

            emp_data_dict = EmpiricalDist(data_samples, N_qubits)

            stein_score_array_approx_identity = IdentitySteinScore(data_samples, kernel_type, eta, stein_sigma)
            # print('The Identity Score matrix is:\n' , stein_score_array_approx_identity)

            stein_score_array_approx_spectral =  SpectralSteinScore(data_samples, data_samples, J, stein_sigma)
            # print('The Spectral Score matrix is:\n' , stein_score_array_approx_spectral)

            stein_score_array_exact_mass = MassSteinScore(data_samples, data_dict)
            # print('\nThe Exact Score matrix is:\n', stein_score_array_exact_mass)

            stein_score_array_approx_mass = MassSteinScore(data_samples, emp_data_dict)
            # print('\nThe Approx Score matrix using empirical density is:\n', stein_score_array_approx_mass)

            spectral_exact_diff[qubit_index, sample_index] = ComputeScoreDifference(stein_score_array_approx_spectral, stein_score_array_exact_mass, 'Frobenius')
            identity_exact_diff[qubit_index, sample_index] = ComputeScoreDifference(stein_score_array_approx_identity, stein_score_array_exact_mass, 'Frobenius')
            mass_exact_diff[qubit_index, sample_index] = ComputeScoreDifference(stein_score_array_approx_mass, stein_score_array_exact_mass, 'Frobenius')

            print('Difference between exact and spectral method is:', spectral_exact_diff[qubit_index, sample_index])
            print('Difference between exact and identity method is:', identity_exact_diff[qubit_index, sample_index])
            print('Difference between exact and density method is:', mass_exact_diff[qubit_index, sample_index])
    return  spectral_exact_diff, identity_exact_diff, mass_exact_diff, N_data_samples, N_qubits_list

max_qubits = 9
eta = 0.01
# J = 4
def PlotScoreGivenNumberSamples(max_qubits, N_samples, eta):
    spectral_exact_diff, identity_exact_diff, mass_exact_diff, N_data_samples, N_qubits_list = CheckScoreApproximationDifference(max_qubits, eta)

    fig, ax = plt.subplots()  
    spectral_exact_diff_plot = np.zeros((len(N_qubits_list)), dtype = int)
    identity_exact_diff_plot = np.zeros((len(N_qubits_list)), dtype = int)
    mass_exact_diff_plot = np.zeros((len(N_qubits_list)), dtype = int)
    for qubit_index in range(0, len(N_qubits_list)):
        spectral_exact_diff_plot[qubit_index]   = spectral_exact_diff[qubit_index, N_data_samples.index(N_samples)]
        identity_exact_diff_plot[qubit_index]   = identity_exact_diff[qubit_index, N_data_samples.index(N_samples)]
        mass_exact_diff_plot[qubit_index]       = mass_exact_diff[qubit_index,      N_data_samples.index(N_samples)] 


        ax.plot(spectral_exact_diff_plot,  '%so' %('r'), label ='Spectral Score')
        ax.plot(identity_exact_diff_plot,  '%s+' %('b'), label ='Identity Score')
        ax.plot(mass_exact_diff_plot,  '%sx' %('g'), label ='Mass Score')
        ax.set_title("Frobenius Norm of Score Matrix using %i samples," %(N_samples)) 
        ax.set_xlabel("Number of Qubits")
        ax.set_ylabel("Frobenius Norm")
                
        ax.set_xticks(np.arange(len(N_qubits_list)))
        ax.set_xticklabels(N_qubits_list)
        ax.legend(('Spectral Score','Identity Score', 'Mass Score'))
        plt.show()

    return
# PlotScoreGivenNumberSamples(max_qubits, 10, eta)
# PlotScoreGivenNumberSamples(max_qubits, 20, eta)

def ComputeWeightedKernel(device_params, kernel_dict, data_samples_list, data_probs, sample_list_1, sample_list_2, stein_params, *argsv):
    '''This kernel computes the weighted kernel for all samples from the two distributions sample_list_1, sample_list_2'''

    device_name = device_params[0]
    as_qvm_value = device_params[1]
    qc = get_qc(device_name, as_qvm = as_qvm_value)
    qubits = qc.qubits()
    N_qubits = len(qubits)

    # sample_kernel_dict = ComputeSampleKernelDict(N_qubits, kernel_dict, sample_list_1, sample_list_2)
    # sample_kernel_matrix = ConvertKernelDictToMatrix(sample_kernel_dict,  N_samples_1, N_samples_2)
    
    delta_x_kernel_dict, delta_y_kernel_dict, kernel_shifted_trace = ComputeDeltaTerms(N_qubits, sample_list_1, sample_list_2, kernel_dict)

    delta_x_matrix_slices = DeltaDictsToMatrix(N_qubits, delta_x_kernel_dict,  sample_list_1, sample_list_2)
    delta_y_matrix_slices = DeltaDictsToMatrix(N_qubits, delta_y_kernel_dict,  sample_list_1, sample_list_2)

    #Parameters required for computing the Stein Score
    score_approx        = stein_params[0]
    J                   = stein_params[1]
    chi                 = stein_params[2]
    stein_kernel_choice = stein_params[3]
    stein_sigma         = stein_params[4]


    if (score_approx == 'Exact_Score'):
        stein_score_matrix_1 = MassSteinScore(sample_list_1, data_probs)
        stein_score_matrix_2 = MassSteinScore(sample_list_2, data_probs)
    elif (score_approx == 'Identity_Score'):
        stein_score_matrix_1 = IdentitySteinScore(data_samples_list, stein_kernel_choice, chi, stein_sigma)
        stein_score_matrix_2 = IdentitySteinScore(data_samples_list, stein_kernel_choice, chi, stein_sigma)
    elif (score_approx == 'Spectral_Score'):
        #compute score matrix using spectral method for all samples, x and y according to the 
        stein_score_matrix_1 = SpectralSteinScore(sample_list_1, data_samples_list, J, stein_sigma)
        stein_score_matrix_2 = SpectralSteinScore(sample_list_2, data_samples_list, J, stein_sigma)

    else: raise IOError('Please enter \'Exact_Score\', \'Identity_Score\' or \'Spectral_Score\' for score_approx')

    # stein_score_dict1, stein_score_dict_samples1 = SteinMatrixtoDict(stein_score_matrix_1, sample_list_1)
    # stein_score_dict2, stein_score_dict_samples2 = SteinMatrixtoDict(stein_score_matrix_2, sample_list_2)

    weighted_kernel = []
    sample_index1 = 0

    for sample1 in sample_list_1:
        sample_index2 = 0
        for sample2 in sample_list_2:
            if sample_index1 == sample_index2:
                if 'same' not in argsv: #if 'same' in optional args, do not contributions that are computed with themselves
                    first_term = np.dot(np.transpose(stein_score_matrix_1[sample_index1][:]), np.dot(kernel_dict[(sample1, sample2)],stein_score_matrix_2[sample_index2][:]))
                    second_term = - np.dot(np.transpose(stein_score_matrix_1[sample_index1][:]), delta_y_matrix_slices[(sample1, sample2)])
                    third_term = - np.dot(np.transpose(delta_x_matrix_slices[(sample1, sample2)]), stein_score_matrix_2[sample_index2][:])
                    fourth_term = kernel_shifted_trace[(sample1, sample2)]
                    weighted_kernel.append(first_term + second_term + third_term + fourth_term) #if the same sample appears it is overwritten
            else:
                first_term = np.dot(np.transpose(stein_score_matrix_1[sample_index1][:]), np.dot(kernel_dict[(sample1, sample2)],stein_score_matrix_2[sample_index2][:]))
                second_term = - np.dot(np.transpose(stein_score_matrix_1[sample_index1][:]), delta_y_matrix_slices[(sample1, sample2)])
                third_term = - np.dot(np.transpose(delta_x_matrix_slices[(sample1, sample2)]), stein_score_matrix_2[sample_index2][:])
                fourth_term = kernel_shifted_trace[(sample1, sample2)]
                weighted_kernel.append(first_term + second_term + third_term + fourth_term) #if the same sample appears it is overwritten

            sample_index2 += 1
        sample_index1 += 1


    return weighted_kernel



