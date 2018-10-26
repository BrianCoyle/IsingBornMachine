import numpy as np
import matplotlib.pyplot as plt

from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler, EmpiricalDist
from classical_kernel import GaussianKernel
from mmd_kernel import KernelCircuit, KernelComputation, EncodingFunc
from mmd_sampler2 import MMDKernel, MMDGrad, MMDCost
from mmd import MMDTrain
from file_operations_in import DataDictFromFile

import sys

def PlotMMD(N, N_v, N_epochs, \
            J_i, b_i, g_x_i, g_y_i, \
            learning_rate, approx, kernel_type,\
            data_samples, data_exact_dict, \
            N_data_samples, N_born_samples, N_bornplus_samples, N_bornminus_samples, N_kernel_samples,\
            plot_colour, weight_sign):
    N_h = N - N_v
    #Output MMD Loss function and parameter values, for given number of training samples
    L, J, b, gamma_x, gamma_y, born_probs_list, empirical_probs_dict  = MMDTrain(N, N_h, N_v,\
                                                                                J_i, b_i, g_x_i, g_y_i, \
                                                                                N_epochs, N_data_samples, N_born_samples,\
                                                                                N_kernel_samples, \
                                                                                data_samples, data_exact_dict, \
                                                                                kernel_type, learning_rate, approx, weight_sign)
    if (approx == 'Sampler'):
        if kernel_type == 'Quantum':
            plt.plot(L, '%so-' %(plot_colour), label ='%i Data Samples,  %i Born Samples for a %s kernel with %i Measurements.' \
                                        %(N_data_samples, N_born_samples, kernel_type[0], N_kernel_samples))
        else:
            plt.plot(L, '%so-' %(plot_colour), label ='%i Data Samples,  %i Born Samples with %s kernel.' \
                            %(N_data_samples, N_born_samples, kernel_type[0]))
    elif (approx == 'Exact'):
        if kernel_type == 'Quantum':
            plt.plot(L, '%so-' %(plot_colour), label ='Exact %s kernel with $\eta$  = %.4f.' \
                                        %(kernel_type[0], learning_rate))
        else:
            plt.plot(L, '%so-' %(plot_colour), label ='Exact %s kernel with $\eta$   = %.4f.' \
                                        %(kernel_type[0], learning_rate))

  
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel("Epochs")
    plt.ylabel("MMD Loss")
    plt.title("MMD Loss for %i qubits" % N_v)

    return L, J, b, gamma_x, gamma_y, born_probs_list, empirical_probs_dict

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

def DataImport(approx, N_qubits, N_data_samples):
    if (approx == 'Sampler'):
        data_samples_orig = np.loadtxt('Data_%iNv_%iSamples' % (N_qubits, N_data_samples), dtype = str)
        data_samples = SampleListToArray(data_samples_orig, N_qubits)
        
        empirical_dist_dict = DataDictFromFile(N_qubits, N_data_samples)

        empirical_dist_dict =  DataDictFromFile(N_qubits, N_data_samples)
        data_exact_dict = DataDictFromFile(N_qubits, 'infinite')
      

    elif (approx == 'Exact'):
        data_exact_dict =  DataDictFromFile(N_qubits, 'infinite')
        data_samples = []

    return data_samples, data_exact_dict 
#DataImport('Sampler', 2, 10000)

def PrintFinalParamsToFile(J, b, L, N_v, kernel_type, N_born_samples, N_epochs, N_data_samples, learning_rate):
    print("THIS IS THE DATA FOR MMD WITH %i VISIBLE QUBITS, WITH %s KERNEL, %i SAMPLES FROM THE BORN MACHINE,\
                %i DATA SAMPLES, %i NUMBER OF EPOCHS AND LEARNING RATE = %.3f" \
                        %(N_v, kernel_type[0], N_born_samples, N_epochs, N_data_samples, learning_rate))
    for epoch in range(0, N_epochs-1):
        print('The weights for Epoch', epoch ,'are :', J[:,:,epoch], '\n')
        print('The biases for Epoch', epoch ,'are :', b[:,epoch], '\n')
        print('MMD Loss for Epoch', epoch ,'is:', L[epoch], '\n')

    return
