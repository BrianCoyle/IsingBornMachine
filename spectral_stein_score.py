import numpy as np
from file_operations_in import ConvertKernelDictToArray, DataImport
from numpy import linalg as LA
from classical_kernel import GaussianKernelArray
from auxiliary_functions import ToString, StringToArray, EmpiricalDist, ShiftString

def LargestEigValsVecs(kernel_array, J):
    '''This function returns the J^th largest eigenvalues and eigenvectors
        of the kernel matrix to compute score using spectral method'''

    kernel_eigvals, kernel_eigvecs = LA.eig(kernel_array)

    #put all eigenvalues and eigenvectors in dictionary
    eig_dict = {}
    eig_iterator = 0
    for eigenvalue in kernel_eigvals:
        eig_dict[eigenvalue] = kernel_eigvecs[:, eig_iterator]
        eig_iterator += 1
    
    #Put eigenvectors in dictionary corresponding to J^th largest eigenvalues
    largest_eigs = list(sorted(eig_dict.keys())[::-1])[0:J]
    largest_eigs_dict = {}
    for eigenvalue in largest_eigs:
        largest_eigs_dict[eigenvalue] = eig_dict[eigenvalue]
   
    return largest_eigs_dict


def ComputeNystromEigenvectorsSingleSample(new_sample, samples, kernel_array_all_samples, J, sigma):
    '''This function computes the approximate eigenvectors psi of the 
    weighed kernel using the Nystrom method, for a given sample, x'''
    psi = np.zeros((J)) #initialise numpy array for J^th approximate eigenvectors

    largest_eigs_dict = LargestEigValsVecs(kernel_array_all_samples, J)
    D = len(samples)
    eigvals_list = list(largest_eigs_dict.keys())
    kernel_array_single_sample = GaussianKernelArray(new_sample, samples, sigma) #Compute kernel matrix for a sample, with all others
    for j in range(0, J):
        temp = np.real((np.sqrt(D)/eigvals_list[j])*np.dot(largest_eigs_dict[eigvals_list[j]], np.transpose(kernel_array_single_sample)))
        psi[j] = temp
    return psi

def ComputeNystromEigenvectorsAllSamples(samples, kernel_array_all_samples, J, sigma):
    '''This function computes the set of nystrom eigenvectors for all samples'''
    NystromEigenvectorsAllSamples = []
    for sample in samples:
        NystromEigenvectorsAllSamples.append(ComputeNystromEigenvectorsSingleSample(sample, samples, J,sigma, kernel_array_all_samples))
    return NystromEigenvectorsAllSamples

def SpectralBetaArray(samples, kernel_array_all_samples, J, sigma):

    N_qubits = len(samples[0])
    #List of arrays of Nystrom eigenvectors, for all samples
    psi_all_samples = ComputeNystromEigenvectorsAllSamples(samples, kernel_array_all_samples, J, sigma)
    D = len(samples)
    #initialise array to be summed over with each index being 
    # (shifted bit, Nystrom eigenvec index, sample index)
    beta_summand = np.zeros((N_qubits, D, J))

    for bit_index in range(0, N_qubits):
        for sample_index in range(0, D):
            sample = ToString(samples[sample_index, :])

            shifted_string = ShiftString(sample, bit_index)
            shifted_string_array = StringToArray(shifted_string)

            beta_summand[bit_index, sample_index, :]= psi_all_samples[sample_index][:] \
            -ComputeNystromEigenvectorsSingleSample(shifted_string_array,\
                                                    samples,\
                                                    kernel_array_all_samples,\
                                                    J, sigma)

    beta = (1/D)*beta_summand.sum(axis = 1)
    return beta

def SpectralSteinScoreSingleSample(new_sample, samples, kernel_array_all_samples, J, sigma):
    '''Compute Stein Score using Spectral method'''
    beta = SpectralBetaArray(samples, kernel_array_all_samples, J, sigma)
    psi = ComputeNystromEigenvectorsSingleSample(new_sample,\
                                                    samples,\
                                                    kernel_array_all_samples,\
                                                    J, sigma)
    
    return np.dot(beta, psi)


def SpectralSteinScore(samples, J, sigma):
    '''This function compute the Approximate Stein Score matrix for all samples '''

    kernel_array = GaussianKernelArray(samples, samples, sigma)
    N_qubits = len(samples[0])
    N_samples = len(samples)
    stein_score_array_spectral = np.zeros((N_samples, N_qubits))
    for sample_index in range(0, N_samples):
        stein_score_array_spectral[sample_index][:] = \
            SpectralSteinScoreSingleSample(samples[sample_index], samples, J, sigma, kernel_array)

    return stein_score_array_spectral
