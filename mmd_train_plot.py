import numpy as np
import matplotlib.pyplot as plt

from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler
from classical_kernel import GaussianKernel
from mmd import MMDTrain, MMDKernelforGradExact, MMDKernelforGradSampler,  MMDGradSampler,MMDGradExact,  KernelComputation, EncodingFunc, MMDCost, MMDCostExact
from file_operations_out import DataDictFromFile

import sys

def PlotMMD(N, N_v, N_epochs, \
            J_i, b_i, g_x_i, g_y_i, \
            learning_rate, approx, kernel_type,\
            data_samples, data_exact_dict, \
            N_data_samples, N_born_samples, N_bornplus_samples, N_bornminus_samples, N_kernel_samples,\
            plot_colour):
    N_h = N - N_v
    #Output MMD Loss function and parameter values, for given number of training samples
    L, J, b, gamma_x, gamma_y  = MMDTrain(N, N_h, N_v,\
                                            J_i, b_i, g_x_i, g_y_i, \
                                            N_epochs, N_data_samples, N_born_samples,\
                                            N_bornplus_samples,  N_bornminus_samples, N_kernel_samples, \
                                            data_samples, data_exact_dict, \
                                            kernel_type, learning_rate, approx)
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

    return L, J, b, gamma_x, gamma_y

def DataImport(approx, N_v, N_data_samples):
    if (approx == 'Sampler'):
        data_samples = np.loadtxt('Data_%iNv_%iSamples' % (N_v, N_data_samples), dtype = float)
        data_exact_dict = {}
        print("The Training Data over", N_v, "Visible qubits is\n", data_samples)
    elif (approx == 'Exact'):
        data_exact_dict = DataDictFromFile(N_v)
        data_samples = []
        print("The Training Data Dict over", N_v, "Visible qubits is\n", data_exact_dict)
    return data_samples, data_exact_dict

def PrintParamsToFile(J, b, L, N_v, kernel_type, N_born_samples, N_epochs, N_data_samples, learning_rate):
    print("THIS IS THE DATA FOR MMD WITH %i VISIBLE QUBITS, WITH %s KERNEL, %i SAMPLES FROM THE BORN MACHINE,\
                %i DATA SAMPLES, %i NUMBER OF EPOCHS AND LEARNING RATE = %.3f" \
                        %(N_v, kernel_type[0], N_born_samples, N_epochs, N_data_samples, learning_rate))
    for epoch in range(0, N_epochs-1):
        print('The weights for Epoch', epoch ,'are :', J[:,:,epoch], '\n')
        print('The biases for Epoch', epoch ,'are :', b[:,epoch], '\n')
        print('MMD Loss for Epoch', epoch ,'is:', L[epoch], '\n')

    return
