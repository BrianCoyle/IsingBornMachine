from pyquil.quil import Program
import numpy as np
from pyquil.api import get_qc
from numpy.linalg import inv

from train_generation import DataSampler
from classical_kernel import GaussianKernelArray, GaussianKernel
from file_operations_in import KernelDictFromFile, DataImport

from auxiliary_functions import ShiftString, ToString, EmpiricalDist
from spectral_stein_score import SpectralSteinScore

import matplotlib.pyplot as plt


def ComputeInverseTerm(kernel_array, N_samples, chi):
    '''
    This function computes the inverse matrix required by the Stein Score Approximator
    '''
    return inv(kernel_array - chi*np.identity(N_samples))
    
def ComputeKernelShift(samples, stein_kernel, stein_sigma):
    '''
    This kernel will not be the same as the one used in the MMD, it will only be computed
    between all samples from distribution P, with every sample from the SAME distribution P
    '''
    N_samples = len(samples)
    N_qubits = len(samples[0])

    shifted_kernel_for_score = np.zeros((N_samples, N_samples, N_qubits))

    for sample_1_index in range(0, N_samples):
        for sample_2_index in range(0, N_samples):
            for qubit in range(0, N_qubits):

                sample_1        = ToString(samples[sample_1_index])
                sample_2        = ToString(samples[sample_2_index])
                shiftedstring2  = ShiftString(sample_2, qubit)
                if stein_kernel == 'Stein_Gaussian':
                    shifted_kernel_for_score[sample_1_index][sample_2_index][qubit]  = \
                        GaussianKernel(sample_1, sample_2, stein_sigma) - GaussianKernel(sample_1, shiftedstring2, stein_sigma)
            
        
    shifted_kernel_array = shifted_kernel_for_score.sum(axis = 1)/N_samples

    return shifted_kernel_array

def IdentitySteinScore(samples, kernel_choice, chi, stein_sigma):
    '''This function computes the Stein Score matrix for all samples, based
    on the method of inverting Stein's identity'''

    N_samples = len(samples)

    #compute kernel matrix between all samples
    if kernel_choice == 'Gaussian':
        kernel_array = GaussianKernelArray(samples, samples, stein_sigma)
    else: raise IOError('\'kernel_choice\' must be \'Gaussian\' ')
    #Compute inverse term in Stein score approximation
    inverse = ComputeInverseTerm(kernel_array, N_samples, chi)
    #Compute shifted kernel term in Stein Score Approximation
    shifted_kernel_matrix  = ComputeKernelShift(samples, kernel_choice, stein_sigma)

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
        shifted_string = ToString(ShiftString(sample_string, bit_index))
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
        stein_score_mass_array[sample_index] = MassSteinScoreSingleSample(samples[sample_index], data_dict)
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
    stein_sigma = [0.25, 10, 100]
    N_data_samples = [10, 20]
    
    # N_data_samples = [10, 100, 200, 300, 400]
    kernel_choice = 'Gaussian'
    data_type = 'Classical_Data'
    
    spectral_exact_diff, identity_exact_diff, mass_exact_diff = [np.zeros((len(N_qubits_list), len(N_data_samples))) for _ in range(3)]

    for qubit_index in range(0, len(N_qubits_list)):
        N_qubits = N_qubits_list[qubit_index]
        for sample_index in range(0, len(N_data_samples)):
            J = N_qubits + 2
            N_samples = N_data_samples[sample_index]
            data_samples, data_dict = DataImport(data_type, N_qubits, N_samples)

            emp_data_dict = EmpiricalDist(data_samples, N_qubits)

            stein_score_array_approx_identity = IdentitySteinScore(data_samples, kernel_choice, eta, stein_sigma)
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




