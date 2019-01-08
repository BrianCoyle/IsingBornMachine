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
    largest_eigvals = list(sorted(eig_dict.keys(), reverse = True))[0:J]
    largest_eigvecs = []
    for eigenvalue in largest_eigvals:
        largest_eigvecs.append(eig_dict[eigenvalue])
   
    return largest_eigvals, largest_eigvecs


def NystromEigenvectorsSingleSample(new_sample, samples, largest_eigvals, largest_eigvecs, J, stein_sigma):
    '''This function computes the approximate eigenvectors psi of the 
    weighed kernel using the Nystrom method, for a given sample, x'''
    psi = np.zeros((J)) #initialise numpy array for J^th approximate eigenvectors
    M = len(samples)
    # eigvals_list = list(largest_eigs_list.keys())
    # eigvecs = np.array(list(largest_eigs_list.values()))
    kernel_array_single_sample = GaussianKernelArray(new_sample, samples, stein_sigma) #Compute kernel matrix for a sample, with all others
    for j in range(0, J):
        psi[j] = np.real((np.sqrt(M)/largest_eigvals[j])*np.dot(largest_eigvecs[j], np.transpose(kernel_array_single_sample)))

    return psi

def NystromEigenvectorsAllSamples(samples, largest_eigvals, largest_eigvecs, J, stein_sigma):
    '''This function computes the set of nystrom eigenvectors for all samples'''
    NystromEigenvectorsAllSamples = []
    for sample in samples:
        NystromEigenvectorsAllSamples.append(NystromEigenvectorsSingleSample(sample, samples, largest_eigvals, largest_eigvecs, J, stein_sigma))
    return NystromEigenvectorsAllSamples

def SpectralBetaArray(samples, largest_eigvals, largest_eigvecs, J, stein_sigma):

    N_qubits = len(samples[0])
    #List of arrays of Nystrom eigenvectors, for all samples
    psi_all_samples = NystromEigenvectorsAllSamples(samples, largest_eigvals, largest_eigvecs, J, stein_sigma)
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
            -NystromEigenvectorsSingleSample(shifted_string_array,\
                                                    samples,\
                                                    largest_eigvals, largest_eigvecs,\
                                                    J, stein_sigma)

    beta = (1/D)*beta_summand.sum(axis = 1)
    return beta

def SpectralSteinScoreSingleSample(new_sample, samples, largest_eigvals, largest_eigvecs, J, stein_sigma):
    '''Compute Stein Score using Spectral method'''
    beta = SpectralBetaArray(samples, largest_eigvals, largest_eigvecs, J, stein_sigma)
    psi = NystromEigenvectorsSingleSample(new_sample,\
                                            samples,\
                                            largest_eigvals, largest_eigvecs,\
                                            J, stein_sigma)

    return np.dot(beta, psi)

def SpectralSteinScore(samples1, samples2, J, stein_sigma):
    '''This function compute the Approximate Stein Score matrix for all samples '''
    #samples2 are from the distribution that we want the score function for
    #samples1 are the samples 
    kernel_array_all_samples = GaussianKernelArray(samples2, samples2, stein_sigma)
    largest_eigvals, largest_eigvecs = LargestEigValsVecs(kernel_array_all_samples, J)

    N_qubits = len(samples2[0])
    N_samples = len(samples2)

    stein_score_array_spectral = np.zeros((N_samples, N_qubits))
    for sample_index in range(0, N_samples):
        stein_score_array_spectral[sample_index][:] = \
            SpectralSteinScoreSingleSample(samples2[sample_index][:], samples2, largest_eigvals, largest_eigvecs, J, stein_sigma)
    return stein_score_array_spectral

