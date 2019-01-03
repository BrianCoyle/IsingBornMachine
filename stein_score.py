from pyquil.quil import Program
import numpy as np
from pyquil.api import get_qc
from numpy.linalg import inv

from train_generation import TrainingData, DataSampler
from classical_kernel import GaussianKernelArray, GaussianKernel
from file_operations_in import KernelDictFromFile, DataImport

from auxiliary_functions import ShiftString, SampleArrayToList, StringToList, ConvertToString, ToString, EmpiricalDist
from spectral_stein_score import SpectralSteinScore

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
    
def ComputeKernelShift(samples, kernel_choice, sigma):
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
                    GaussianKernel(sample_1, sample_2, sigma) -\
                    GaussianKernel(sample_1, ShiftString(sample_2, qubit), sigma)
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

def IdentitySteinScore(samples, kernel_type, chi, sigma):
    '''This function computes the Stein Score matrix for all samples, based
    on the method of inverting Stein's identity'''

    N_samples = len(samples)

    #compute kernel matrix between all samples
    kernel_array = GaussianKernelArray(samples, samples, sigma)

    #Compute inverse term in Stein score approximation
    inverse = ComputeInverseTerm(kernel_array, N_samples, chi)
    #Compute shifted kernel term in Stein Score Approximation
    shifted_kernel_matrix  = ComputeKernelShift(samples, kernel_type, sigma)

    #Compute Approximate kernel
    stein_score_matrix_identity = N_samples*np.dot(inverse, np.transpose(shifted_kernel_matrix))

    return stein_score_matrix_identity


def MassSteinScoreSingleSample(sample, data_dict):
    '''This computes the exact Stein Score function in the discrete case for a single 
    sample which is a 1D numpy array, based on probability mass data_dict'''
    if sample.ndim != 1:
        raise TypeError('\'sample\' must be a 1D numpy array')

    N_qubits = len(sample)
    sample_string = ToString(sample)
    stein_score_sample_mass = np.zeros((N_qubits))
    for bit_index in range(0, N_qubits):
        shifted_string = ShiftString(sample_string, bit_index)
        stein_score_sample_mass[bit_index] = 1 - data_dict[shifted_string]/data_dict[sample_string]

    return stein_score_sample_mass

def MassSteinScore(samples, data_dict):
    '''This computes the Stein Matrix for all samples, based on probability mass'''
    if type(samples) is not np.ndarray:
        raise TypeError('\'samples\' must be a numpy array')

    N_samples = len(samples)
    N_qubits  = len(samples[0])
    stein_score_array_mass = np.zeros((N_samples, N_qubits))

    for sample_index in range(0, N_samples):
        stein_score_array_mass[sample_index][:] =\
                 MassSteinScoreSingleSample(samples[sample_index], data_dict)

    return stein_score_array_mass


def ComputeScoreDifference(array_1, array_2, norm_type):
    '''This function computes either the Frobenius Norm, Infinity norm or a simple sum difference
        between the two arrays'''

    if (norm_type is 'Frobenius'):
        Norm = np.linalg.norm((array_1 - array_2), ord = None)
    elif (norm_type is 'Infinity'):
        Norm = np.linalg.norm(array_1 - array_2, ord = np.inf)
    elif (norm_type is 'Difference'):
        #Use simple diffence as sum of all differences between two entries
        Norm = (np.abs(array_1 - array_2)).sum()
    else: raise IOError('\'norm_type\' must be \'Frobenius\', \'Infinity\', or \'Difference\' ')
    return Norm


N_qubits = 2
sigma = [0.1, 10, 100]

N_kernel_samples = 100
N_data_samples = 100
kernel_type = 'Gaussian'
chi = 0.01
J = 7

# data_samples, data_dict = DataImport('Sampler', N_qubits, N_data_samples, 'Exact_Stein')
# print(data_dict)
# emp_data_dict = EmpiricalDist(data_samples, N_qubits)
# print(emp_data_dict)



# stein_score_array_approx_identity = IdentitySteinScore(data_samples, kernel_type, chi, sigma)
# print('The Identity Score matrix is:\n' , stein_score_array_approx_identity)

# stein_score_array_approx_spectral =  SpectralSteinScore(data_samples, J, sigma)
# print('The Spectral Score matrix is:\n' , stein_score_array_approx_spectral)

# stein_score_array_exact_mass = MassSteinScore(data_samples, data_dict)
# print('\nThe Exact Score matrix is:\n', stein_score_array_exact_mass)

# stein_score_array_approx_mass = MassSteinScore(data_samples, emp_data_dict)
# print('\nThe Approx Score matrix using empirical density is:\n', stein_score_array_approx_mass)

# spectral_exact_diff = ComputeScoreDifference(stein_score_array_approx_spectral, stein_score_array_exact_mass, 'Difference')
# identity_exact_diff = ComputeScoreDifference(stein_score_array_approx_identity, stein_score_array_exact_mass, 'Difference')
# mass_exact_diff = ComputeScoreDifference(stein_score_array_approx_mass, stein_score_array_exact_mass, 'Difference')

# print('Difference between exact and spectral method is:', spectral_exact_diff)
# print('Difference between exact and identity method is:', identity_exact_diff)
# print('Difference between exact and density method is:', mass_exact_diff)





def ComputeWeightedKernel(device_params, kernel_dict, data_samples_list, data_probs, sample_list_1, sample_list_2, score_approx, chi):
    '''This kernel computes the weighted kernel for all samples from the distribution test_samples'''

    device_name = device_params[0]
    as_qvm_value = device_params[1]
    qc = get_qc(device_name, as_qvm = as_qvm_value)
    qubits = qc.qubits()
    N_qubits = len(qubits)

    N_samples_1 = len(sample_list_1)
    N_samples_2 = len(sample_list_2)

    sample_kernel_dict = ComputeSampleKernelDict(N_qubits, kernel_dict, sample_list_1, sample_list_2)
    sample_kernel_matrix = ConvertKernelDictToMatrix(sample_kernel_dict,  N_samples_1, N_samples_2)
    
    delta_x_kernel_dict, delta_y_kernel_dict, kernel_shifted_trace = ComputeDeltaTerms(N_qubits, sample_list_1, sample_list_2, kernel_dict)

    delta_x_matrix_slices = DeltaDictsToMatrix(N_qubits, delta_x_kernel_dict,   sample_list_1, sample_list_2)
    delta_y_matrix_slices = DeltaDictsToMatrix(N_qubits, delta_y_kernel_dict,   sample_list_1, sample_list_2)

    if (score_approx == 'Exact_Score'):
        stein_score_matrix_1 = ExactSteinScore(sample_list_1, data_probs)
        stein_score_matrix_2 = ExactSteinScore(sample_list_2, data_probs)
    elif (score_approx == 'Approx_Score'):
        stein_score_matrix_1 = ApproxSteinScoreIdentity(sample_kernel_matrix, kernel_dict, sample_list_1, chi)
        stein_score_matrix_2 = ApproxSteinScoreIdentity(sample_kernel_matrix, kernel_dict, sample_list_2, chi)

    else: raise IOError('Please enter \'Exact_Score\' or \'Approx_Score\' for score_approx')

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



