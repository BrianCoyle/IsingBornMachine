import numpy as np
import matplotlib.pyplot as plt

from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler
from classical_kernel import GaussianKernel
from mmd_kernel import KernelCircuit, KernelComputation, EncodingFunc
from mmd_sampler2 import MMDKernel, MMDGrad, MMDCost
from cost_function_train import TrainBorn
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
    
    #Output MMD Loss function and parameter values, for given number of training samples
    L_stein, L_var, L_mmd, J, b, gamma_x, gamma_y, born_probs_list, empirical_probs_dict = TrainBorn(N, N_v, cost_func,\
                                                                            J_i, b_i, g_x_i, g_y_i, \
                                                                            N_epochs, N_data_samples, N_born_samples,\
                                                                            N_kernel_samples, \
                                                                            data_samples, data_exact_dict, \
                                                                            kernel_type, learning_rate, approx, score_approx, weight_sign)
	
    if (approx == 'Sampler'):                       
        if kernel_type == 'Quantum':
            plt.plot(L_mmd,  '%so-' %(plot_colour[0]), label ='MMD, %i Data Samples,  %i Born Samples for a %s kernel with %i Measurements.' \
                                        %(N_data_samples, N_born_samples, kernel_type[0], N_kernel_samples))
            # plt.plot(L_stein, 'bo-' , label ='Stein, %i Data Samples,  %i Born Samples for a %s kernel with %i Measurements.' \
            #                             %(N_data_samples, N_born_samples, kernel_type[0], N_kernel_samples))
            plt.plot(L_var, '%so-' %(plot_colour[1]), label ='TV, %i Data Samples,  %i Born Samples for a %s kernel with %i Measurements.' \
                                        %(N_data_samples, N_born_samples, kernel_type[0], N_kernel_samples))
        else:
            plt.plot(L_mmd,  '%so-' %(plot_colour[0]), label ='MMD, %i Data Samples,  %i Born Samples for a %s kernel.' \
                                        %(N_data_samples, N_born_samples, kernel_type[0]))
            # plt.plot(L_stein, 'bo-' , label ='Stein, %i Data Samples,  %i Born Samples for a %s kernel.' \
            #                             %(N_data_samples, N_born_samples, kernel_type[0]))
            plt.plot(L_var, '%so-' %(plot_colour[1]), label ='TV, %i Data Samples,  %i Born Samples for a %s kernel.' \
                                        %(N_data_samples, N_born_samples, kernel_type[0]))
    elif (approx == 'Exact'):
        if kernel_type == 'Quantum':
            plt.plot(L_mmd, '%so-' %(plot_colour[0]), label ='Exact %s kernel with $\eta$  = %.4f.' \
                                        %(kernel_type[0], learning_rate))
        else:
            plt.plot(L_mmd, '%so-' %(plot_colour[0]), label ='Exact %s kernel with $\eta$   = %.4f.' \
                                    %(kernel_type[0], learning_rate))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel("Epochs")
    plt.ylabel("MMD Loss")
    plt.title("MMD Loss for %i qubits" % N_v)

    return L_stein, L_var, L_mmd , J, b, gamma_x, gamma_y, born_probs_list, empirical_probs_dict


