import numpy as np
import matplotlib.pyplot as plt
from cost_function_train import TrainBorn

from file_operations_in import TrainingDataFromFile
import sys

def CostPlot(qc, N_epochs, initial_params, \
            kernel_type,\
            data_train_test, data_exact_dict, \
            N_samples,\
            cost_func, flag, learning_rate, stein_params, sinkhorn_eps):
            
    N_qubits = len(qc.qubits())
    #Output MMD Loss function and parameter values, for given number of training samples
    loss, circuit_params, born_probs_list, empirical_probs_list = TrainBorn(qc, cost_func,\
                                                                            initial_params, \
                                                                            N_epochs, N_samples, \
                                                                            data_train_test, data_exact_dict, \
                                                                            kernel_type, flag, learning_rate, \
                                                                            stein_params, sinkhorn_eps)
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

###FIXXX
def PlotLoss(cost_func, qc, N_epochs, initial_params, \
            kernel_type,\
            N_samples):
    [N_data_samples,N_born_samples, batch_size, N_kernel_samples] = N_samples

    N_qubits  = len(qc.qubits())

    loss, _ = TrainingDataFromFile(cost_func, qc, kernel_type, N_kernel_samples, N_data_samples, N_born_samples, batch_size, N_epochs)  

   
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
        elif (cost_func == 'TV'):
            plt.plot(loss['TV'], '%so-' %(plot_colour[1]), label ='TV, %i Data Samples,  %i Born Samples for a %s kernel.' \
                                    %(N_samples[0], N_samples[1], kernel_type[0]))
   
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss for %i qubits" % N_qubits)

    return
