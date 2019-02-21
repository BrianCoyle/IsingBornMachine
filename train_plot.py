import numpy as np
import matplotlib.pyplot as plt

from file_operations_in import TrainingDataFromFile
import sys

def CostPlot(N_qubits, kernel_type, data_train_test, N_samples,\
            cost_func, loss, circuit_params, born_probs_list, empirical_probs_list):
            
    plot_colour = ['r', 'b']

    if kernel_type == 'Quantum':
        if (cost_func == 'MMD'):
            plt.plot(loss[('MMD', 'Train')],  '%so-' %(plot_colour[0]), label ='MMD, %i Training Points,  %i Born Samples for a %s kernel with %i Measurements.' \
                                        %(len(data_train_test[0]), N_samples[1], kernel_type[0], N_samples[-1]))
            plt.plot(loss[('MMD', 'Test')],  '%sx-' %(plot_colour[0]), label ='MMD, %i Test Points,  %i Born Samples for a %s kernel with %i Measurements.' \
                                        %(len(data_train_test[1]), N_samples[1], kernel_type[0], N_samples[-1]))
        elif (cost_func == 'Stein'):
            plt.plot(loss[('Stein', 'Train')],'%so' %(plot_colour[1]) , label ='Stein, %i Training Points,  %i Born Samples for a %s kernel with %i Measurements.' \
                                    %(len(data_train_test[0]), N_samples[1], kernel_type[0], N_samples[-1]))
            plt.plot(loss[('Stein', 'Test')],'%so-' %(plot_colour[1]) , label ='Stein, %i Test Points,  %i Born Samples for a %s kernel with %i Measurements.' \
                                    %(len(data_train_test[1]), N_samples[1], kernel_type[0], N_samples[-1]))
        elif (cost_func == 'TV'):
            plt.plot(loss['TV'], '%so-' %(plot_colour[1]), label ='TV, %i Data Samples,  %i Born Samples for a %s kernel with %i Measurements.' \
                                    %(N_samples[0], N_samples[1], kernel_type[0],  N_samples[-1]))
    else:
        if (cost_func == 'MMD'):
            plt.plot(loss[('MMD', 'Train')],  '%so-' %(plot_colour[0]), label ='MMD, %i Training Points,  %i Born Samples for a %s kernel.' \
                                        %(len(data_train_test[0]), N_samples[1], kernel_type[0]))
            plt.plot(loss[('MMD', 'Test')],  '%sx-' %(plot_colour[0]), label ='MMD, %i Test Points,  %i Born Samples for a %s kernel.' \
                                        %(len(data_train_test[1]), N_samples[1], kernel_type[0]))
        elif (cost_func == 'Stein'):
            plt.plot(loss[('Stein', 'Train')],'%so-' %(plot_colour[1]) , label ='Stein, %i Training Points,  %i Born Samples for a %s kernel.' \
                                        %(len(data_train_test[0]), N_samples[1], kernel_type[0]))
            plt.plot(loss[('Stein', 'Test')],'%sx-' %(plot_colour[1]) , label ='Stein, %i Test Points,  %i Born Samples for a %s kernel.' \
                                        %(len(data_train_test[1]), N_samples[1], kernel_type[0]))
        elif (cost_func == 'Sinkhorn'):
            plt.plot(loss[('Sinkhorn', 'Train')],'%so-' %(plot_colour[1]) , label ='Sinkhorn, %i Training Points,  %i Born Samples for a Hamming cost.' \
                                        %(len(data_train_test[0]), N_samples[1]))
            plt.plot(loss[('Sinkhorn', 'Test')],'%sx-' %(plot_colour[1]) , label ='Sinkhorn, %i Test Points,  %i Born Samples for a Hamming cost.' \
                                        %(len(data_train_test[1]), N_samples[1]))
        elif (cost_func == 'TV'):
            plt.plot(loss['TV'], '%so-' %(plot_colour[1]), label ='TV, %i Data Samples,  %i Born Samples for a %s kernel.' \
                                    %(N_samples[0], N_samples[1], kernel_type[0]))
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss for %i qubits" % N_qubits)


    plt.show(block=False)
    plt.pause(1)
    plt.close()
    return loss, circuit_params, born_probs_list, empirical_probs_list
