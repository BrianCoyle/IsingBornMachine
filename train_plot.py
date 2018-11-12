import numpy as np
import matplotlib.pyplot as plt

from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler
from classical_kernel import GaussianKernel
from mmd_kernel import KernelCircuit, KernelComputation, EncodingFunc
from mmd_sampler2 import MMDKernel, MMDGrad, MMDCost
from mmd import MMDTrain
from file_operations_in import DataDictFromFile
from auxiliary_functions import EmpiricalDist, SampleListToArray
from stein_discrepancy import SteinTrain
import sys

def Plot(N, N_v, N_epochs, \
            J_i, b_i, g_x_i, g_y_i, \
            learning_rate, approx, kernel_type,\
            data_samples, data_exact_dict, \
            N_data_samples, N_born_samples, N_bornplus_samples, N_bornminus_samples, N_kernel_samples,\
            plot_colour, weight_sign, cost_func, score_approx):
    N_h = N - N_v
    
    if (cost_func == 'MMD'):
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

    elif (cost_func == 'Stein'):
        L, J, b, gamma_x, gamma_y, born_probs_list, empirical_probs_dict  = SteinTrain(N, N_h, N_v,\
                                                                                    J_i, b_i, g_x_i, g_y_i, \
                                                                                    N_epochs, N_data_samples, N_born_samples,\
                                                                                    N_kernel_samples, \
                                                                                    data_samples, data_exact_dict, \
                                                                                    kernel_type, learning_rate, approx, score_approx, weight_sign)
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
        plt.ylabel("Stein Discrepancy")
        plt.title("Stein Discrepancy for %i qubits" % N_v)

    return L, J, b, gamma_x, gamma_y, born_probs_list, empirical_probs_dict


