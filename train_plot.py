import numpy as np
import matplotlib.pyplot as plt

from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler

from cost_function_train import TrainBorn

from file_operations_in import DataDictFromFile
from auxiliary_functions import EmpiricalDist, SampleListToArray
import sys

def CostPlot(N_qubits, N_epochs, initial_params, \
            learning_rate, approx, kernel_type,\
            data_train_test, data_exact_dict, \
            N_samples,\
            plot_colour, weight_sign, cost_func, score_approx, flag, batch_size):
    
    #Output MMD Loss function and parameter values, for given number of training samples
    loss, circuit_params, born_probs_list, empirical_probs_dict = TrainBorn(N_qubits, cost_func,\
                                                                            initial_params, \
                                                                            N_epochs, N_samples, \
                                                                            data_train_test, data_exact_dict, \
                                                                            kernel_type, learning_rate, approx,\
                                                                            score_approx, weight_sign, flag, batch_size)
	
    if (approx == 'Sampler'):                       
        if kernel_type == 'Quantum':
            plt.plot(loss[('MMD', 'Train')],  '%so-' %(plot_colour[0]), label ='MMD, %i Training Points,  %i Born Samples for a %s kernel with %i Measurements.' \
                                        %(len(data_train_test[0]), N_samples[1], kernel_type[0], N_samples[5]))
            plt.plot(loss[('MMD', 'Test')],  '%sx-' %(plot_colour[0]), label ='MMD, %i Test Points,  %i Born Samples for a %s kernel with %i Measurements.' \
                                        %(len(data_train_test[1]), N_samples[1], kernel_type[0], N_samples[5]))
            # plt.plot(loss[('Stein', 'Train')],'%so' %(plot_colour[1]) , label ='Stein, %i Training Points,  %i Born Samples for a %s kernel with %i Measurements.' \
            #                             %(len(data_train_test[0]), N_samples[1], kernel_type[0], N_samples[5]))
            # plt.plot(loss[('Stein', 'Test')],'%so-' %(plot_colour[1]) , label ='Stein, %i Test Points,  %i Born Samples for a %s kernel with %i Measurements.' \
            #                             %(len(data_train_test[1]), N_samples[1], kernel_type[0], N_samples[5]))
            # plt.plot(loss['TV'], '%so-' %(plot_colour[1]), label ='TV, %i Data Samples,  %i Born Samples for a %s kernel with %i Measurements.' \
            #                             %(N_samples[0], N_samples[1], kernel_type[0],  N_samples[5]))
        else:
            plt.plot(loss[('MMD', 'Train')],  '%so-' %(plot_colour[0]), label ='MMD, %i Training Points,  %i Born Samples for a %s kernel.' \
                                        %(len(data_train_test[0]), N_samples[1], kernel_type[0]))
            plt.plot(loss[('MMD', 'Test')],  '%sx-' %(plot_colour[0]), label ='MMD, %i Test Points,  %i Born Samples for a %s kernel.' \
                                        %(len(data_train_test[1]), N_samples[1], kernel_type[0]))
            # plt.plot(loss[('Stein', 'Train')],'%so-' %(plot_colour[1]) , label ='Stein, %i Training Points,  %i Born Samples for a %s kernel.' \
            #                             %(len(data_train_test[0]), N_samples[1], kernel_type[0]))
            # plt.plot(loss[('Stein', 'Test')],'%sx-' %(plot_colour[1]) , label ='Stein, %i Test Points,  %i Born Samples for a %s kernel.' \
            #                             %(len(data_train_test[1]), N_samples[1], kernel_type[0]))
            # plt.plot(loss['TV'], '%so-' %(plot_colour[1]), label ='TV, %i Data Samples,  %i Born Samples for a %s kernel.' \
            #                             %(N_samples[0], N_samples[1], kernel_type[0]))
    elif (approx == 'Exact'):
        if kernel_type == 'Quantum':
            plt.plot(loss['MMD'], '%so-' %(plot_colour[0]), label ='Exact %s kernel with $\eta$  = %.4f.' \
                                        %(kernel_type[0], learning_rate))
        else:
            plt.plot(loss['MMD'], '%so-' %(plot_colour[0]), label ='Exact %s kernel with $\eta$   = %.4f.' \
                                    %(kernel_type[0], learning_rate))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss for %i qubits" % N_qubits)

    return loss, circuit_params, born_probs_list, empirical_probs_dict


