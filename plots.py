import numpy as np
import ast
import sys
import json
from auxiliary_functions import SampleListToArray
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rc('xtick', labelsize=20)     
matplotlib.rc('ytick', labelsize=20) 
import matplotlib.pyplot as plt
from file_operations_in import ReadFromFile, AverageCostsFromFile
from file_operations_out import MakeTrialNameFile, MakeDirectory

'''
This file produces the various plots provided in 'The Born Supremacy: Quantum Advantage and Training of an Ising Born Machine'
'''

####################################################################################################################
# #Compare Costs
###################################################################################################################

def CompareCostFunctions(N_epochs, learning_rate, data_type, data_circuit,
                        N_born_samples, N_data_samples, N_kernel_samples,
                        batch_size, kernel_type, cost_func, qc, score,
                        stein_eigvecs, stein_eta, sinkhorn_eps, runs, comparison, legend = True):
    loss, born_final_probs, data_probs_final = ReadFromFile(N_epochs, learning_rate, data_type, data_circuit,
															N_born_samples, N_data_samples, N_kernel_samples,
															batch_size, kernel_type, cost_func, qc, score,
															stein_eigvecs, stein_eta, sinkhorn_eps, runs)

    if all(x.lower() == 'mmd' for x in cost_func) is True:
        #If all cost functions to be compared are the mmd, 
        plot_colour = ['r*-', 'r+-', 'ro-', 'b*-', 'b+-', 'bo-']
    else:
        plot_colour = ['c-', 'y-', 'g-', 'b-', 'r-', 'm-', 'k-']
    N_trials = len(N_epochs)
    if comparison.lower() == 'probs':

        fig, axs = plt.subplots()
        data_plot_colour = 'k'

        axs.clear()
        #Plot Data
        x = np.arange(len(data_probs_final[0]))
        #Plot MMD
        # axs.bar(x-(0.2*(0+0.5)), born_final_probs[-4].values(), width=0.1, color='%s' %(bar_plot_colour[0]), align='center')

        #Plot other costs
        # print(list(born_final_probs[-2].keys()), x)
        if qc[0][0].lower() == '3':
            bar_plot_colour = ['g', 'b', 'r', 'm']

             #Plot MMD
            axs.bar(x, data_probs_final[0].values(), width=0.1, color= '%s' %data_plot_colour, align='center')

            axs.bar(x-(0.2*(0+0.5)), born_final_probs[-4].values(), width=0.1, color='%s' %(bar_plot_colour[0]), align='center')

            axs.bar(x-(0.2*(0+1)), born_final_probs[-3].values(), width=0.1, color='%s' %(bar_plot_colour[1]), align='center')
            axs.bar(x-(0.2*(0+1.5)), born_final_probs[-2].values(), width=0.1, color='%s' %(bar_plot_colour[2]), align='center')
            axs.bar(x-(0.2*(0+2)), born_final_probs[-1].values(), width=0.1, color='%s' %(bar_plot_colour[3]), align='center')
        
        elif qc[0][0].lower() == '4':
            axs.bar(x, data_probs_final[0].values(), width=0.2, color= '%s' %data_plot_colour, align='center')

            bar_plot_colour = ['g', 'b', 'm']

            #Plot MMD
            axs.bar(x-(0.2*(0+1)), born_final_probs[-4].values(), width=0.2, color='%s' %(bar_plot_colour[0]), align='center')
            axs.bar(x-(0.2*(0+2)), born_final_probs[-3].values(), width=0.2, color='%s' %(bar_plot_colour[1]), align='center')
            axs.bar(x-(0.2*(0+3)), born_final_probs[-1].values(), width=0.2, color='%s' %(bar_plot_colour[2]), align='center')

        axs.set_xlabel("Outcomes", fontsize=20)
        axs.set_ylabel("Probability", fontsize=20)
        if legend == True:
            # axs.set_title(r'Outcome Distributions')
            if qc[0][0].lower() == '3':
                axs.legend(('Data',r'\textsf{MMD}', r'Sinkhorn', r'Exact Stein',  r'Spectral Stein' ), fontsize = 20)
            elif qc[0][0].lower() == '4':
                axs.legend(('Data',r'\textsf{MMD}', r'Sinkhorn',  r'Spectral Stein' ), fontsize = 20)

        axs.set_xticks(range(len(data_probs_final[0])))
        axs.set_xticklabels(list(data_probs_final[0].keys()),rotation=70)

        plt.show()
    elif comparison.lower() == 'tv':
        for trial in range(N_trials):
            #Compute Average losses and errors, over a certain number of runs
            try:
            # if cost_func[trial].lower() != 'stein':
                average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs[trial], learning_rate[trial], data_type[trial], data_circuit[trial],	
                                                                                    N_born_samples[trial], N_data_samples[trial], N_kernel_samples[trial],
                                                                                    batch_size[trial], kernel_type[trial], cost_func[trial], qc[trial], score[trial],
                                                                                    stein_eigvecs[trial], stein_eta[trial], sinkhorn_eps[trial])
                cost_error = np.vstack((lower_error['TV'], upper_error['TV'])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            except:
                pass
                # raise FileNotFoundError('The Average cost could not be found')

            x = np.arange(0, N_epochs[trial]-1, 1)

            if legend == True:
                """WITH LEGEND"""
                if cost_func[trial].lower() == 'mmd':
                    try:
                        plt.errorbar(x, average_loss['TV'], cost_error, None,\
                                                '%s' %(plot_colour[trial]), label =r'%s, %i Data Samples for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                    %(cost_func[trial], N_data_samples[trial], kernel_type[trial][0], learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    except:
                        plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]), label =r'MMD %i Data Samples for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                        %(N_data_samples[trial], kernel_type[trial][0], learning_rate[trial]))
                elif  cost_func[trial].lower() == 'stein':

                    x_stein = np.arange(0, len(average_loss['TV']))
                    if score[trial].lower() == 'exact':
                        try:
                        
                            plt.errorbar(x_stein, average_loss['TV'], cost_error, None,\
                                                        '%s' %(plot_colour[trial]), label =r'Stein %i Data Samples using Exact Score for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                        %(N_data_samples[trial], kernel_type[trial][0], learning_rate[trial]),\
                                                            capsize=1, elinewidth=1, markeredgewidth=2)
                    
                        except:
                            plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]), label =r'Stein %i Data Samples using Exact Score for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                        %(N_data_samples[trial], kernel_type[trial][0], learning_rate[trial]))
                    elif score[trial].lower() == 'spectral':
                       
                        plot_colour  = 'm'

                        plt.plot(loss[trial][('TV')],  '%s' %(plot_colour), label =r'Stein %i Data Samples using Spectral Score for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                        %(N_data_samples[trial], kernel_type[trial][0], learning_rate[trial]))
                elif cost_func[trial].lower() == 'sinkhorn':
                    # plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]), label =r'%s, %i Data $\&$ Born Samples with $\epsilon$ = %.3f.' \
                    #                             %(cost_func[trial], N_data_samples[trial], sinkhorn_eps[trial]))
                    
                    try:
                        plt.errorbar(x, average_loss['TV'], cost_error, None,\
                                                    '%s' %(plot_colour[trial]), label =r'Sinkhorn, %i Data Samples using Hamming Cost, $\eta_{init}$ = %.3f.' \
                                                        %(N_data_samples[trial], learning_rate[trial]),\
                                                        capsize=1, elinewidth=1, markeredgewidth=2)
                    except:
                        plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]), label =r'Stein, %i Data Samples using Spectral Score, $\eta_{init}$ = %.3f.' \
                                                %( N_data_samples[trial], learning_rate[trial]))
            elif legend == False:
                """WITHOUT LEGEND"""
                plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]))
                
        plt.xlabel("Epochs", fontsize=20)
        plt.ylabel("TV", fontsize=20)


        plt.legend(loc='best', prop={'size': 20}).draggable()
        plt.show()       


    elif comparison.lower() == 'cost':
        for trial in range(N_trials):
            try:
                
                average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs[trial], learning_rate[trial], data_type[trial], data_circuit[trial],	
                                                                                    N_born_samples[trial], N_data_samples[trial], N_kernel_samples[trial],
                                                                                    batch_size[trial], kernel_type[trial], cost_func[trial], qc[trial], score[trial],
                                                                                    stein_eigvecs[trial], stein_eta[trial], sinkhorn_eps[trial])
                if cost_func[trial].lower() == 'mmd':
                    train_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('MMD', 'Test')], upper_error[('MMD', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                elif cost_func[trial].lower() == 'stein':
                    train_error = np.vstack((lower_error[('Stein', 'Train')], upper_error[('Stein', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Stein', 'Test')], upper_error[('Stein', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                elif cost_func[trial].lower() == 'sinkhorn':
                    train_error = np.vstack((lower_error[('Sinkhorn', 'Train')], upper_error[('Sinkhorn', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Sinkhorn', 'Test')], upper_error[('Sinkhorn', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            except:
                pass
            if cost_func[trial].lower() == 'mmd':
                plot_colour  = ['c', 'y', 'g']
                # try:
                x_mmd = np.arange(0, len(average_loss['MMD', 'Train']))

                plt.errorbar(x_mmd, average_loss[('MMD', 'Train')], train_error, None,\
                                                '%so' %(plot_colour[trial]), label =r'MMD, %i Train Points using $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(round(N_data_samples[trial]*0.8), kernel_type[trial][0], learning_rate[trial]),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                plt.errorbar(x_mmd, average_loss[('MMD', 'Test')], test_error, None,\
                                                '%s-' %(plot_colour[trial]), label =r'MMD, %i Test Points using $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(N_data_samples[trial] - round(N_data_samples[trial]*0.8),kernel_type[trial][0], learning_rate[trial]),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                # except:
                #     plt.plot(loss[trial][('Stein', 'Train')],  '%so-' %(plot_colour), label =r'Stein, %i Train Points using Exact Score.' \
                #                                 %(round(N_data_samples[trial]*0.8)))
                #     plt.plot(loss[trial][('Stein', 'Test')],  '%s--' %(plot_colour), label =r'Stein, %i Test Points using Exact Score .' \
                                                # %(N_data_samples[trial] - round(N_data_samples[trial]*0.8)))
    
                plt.ylabel(r'MMD Loss $\mathcal{L}_{\mathsf{MMD}}$', fontsize = 20)
            
            elif cost_func[trial].lower() == 'stein':
                if score[trial].lower() == 'exact':
                    plot_colour  = 'r'
                    # try:
                    try:
                        x_stein = np.arange(0, len(average_loss['Stein', 'Train']))

                        plt.errorbar(x_stein, average_loss[('Stein', 'Train')], train_error, None,\
                                                    '%so-' %(plot_colour), label =r'Stein, %i Train Points using Exact Score, $\eta_{init}$ = %.3f.' \
                                                    %(round(N_data_samples[trial]*0.8), learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                        plt.errorbar(x_stein, average_loss[('Stein', 'Test')], test_error, None,\
                                                    '%s-' %(plot_colour), label =r'Stein, %i Test Points using Exact Score, $\eta_{init}$ = %.3f.' \
                                                    %(N_data_samples[trial] - round(N_data_samples[trial]*0.8), learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    except:
                        plt.plot(loss[trial][('Stein', 'Train')],  '%so-' %(plot_colour), label =r'Stein, %i Train Points using Exact Score.' \
                                                    %(round(N_data_samples[trial]*0.8)))
                        plt.plot(loss[trial][('Stein', 'Test')],  '%s--' %(plot_colour), label =r'Stein, %i Test Points using Exact Score .' \
                                                    %(N_data_samples[trial] - round(N_data_samples[trial]*0.8)))
                elif score[trial].lower() == 'spectral':
                    plot_colour  = 'm'
                    

                    plt.plot(loss[trial][('Stein', 'Train')],  '%so-' %(plot_colour), label =r'Stein, %i Train Points using Spectral Score, $\eta_{init}$ = %.3f.' \
                                                %(round(N_data_samples[trial]*0.8), learning_rate[trial] ))

                    plt.plot(loss[trial][('Stein', 'Test')],  '%s-' %(plot_colour), label =r'Stein, %i Test Points using Spectral Score, $\eta_{init}$ = %.3f.' \
                                                %(N_data_samples[trial] - round(N_data_samples[trial]*0.8), learning_rate[trial]))

                plt.ylabel(r'Stein Loss $\mathcal{L}_{\mathsf{SD}}$', fontsize = 20)
            elif cost_func[trial].lower() == 'sinkhorn':
                plot_colour  = 'b'
                try:
                    x_sink = np.arange(0, len(average_loss['Sinkhorn', 'Train']))

                    plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Train')], train_error, None,\
                                                    '%so' %(plot_colour[trial]), label =r'Sinkhorn, %i Train Points using Hamming Cost, $\eta_{init}$ = %.3f.' \
                                                    %(round(N_data_samples[trial]*0.8), learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Test')], test_error, None,\
                                                    '%s-' %(plot_colour[trial]), label =r'Sinkhorn, %i Test Points using Hamming Cost, $\eta_{init}$ = %.3f.' \
                                                    %(N_data_samples[trial] - round(N_data_samples[trial]*0.8), learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                except:
                    plt.plot(loss[trial][('Sinkhorn', 'Train')],  '%so-' %(plot_colour), label =r'Sinkhorn, %i Train Points using Exact Score.' \
                                                %(round(N_data_samples[trial]*0.8)))
                    plt.plot(loss[trial][('Sinkhorn', 'Test')],  '%s--' %(plot_colour), label =r'Sinkhorn, %i Test Points using Exact Score .' \
                                                %(N_data_samples[trial] - round(N_data_samples[trial]*0.8)))
    
                plt.ylabel(r'MMD Loss $\mathcal{L}_{\mathsf{MMD}}$', fontsize = 20)
        plt.xlabel("Epochs", fontsize = 20)
        plt.legend(loc='best', prop={'size': 20}).draggable()
                    
  
        plt.show()
    return

[N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples, batch_size, kernel_type, \
cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, runs] = [[] for _ in range(16)] 


'''THREE QUBITS'''

# N_epochs.append(200)
# learning_rate.append(0.05)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('3q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)    
# sinkhorn_eps.append(0.05)
# runs.append(0)


# N_epochs.append(200)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('3q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.05)
# runs.append(0)


# N_epochs.append(200)
# learning_rate.append(0.15)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('3q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('3q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.05)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Stein')
# qc.append('3q-qvm')
# score.append('Exact') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(4)

# N_epochs.append(125)
# learning_rate.append(0.045)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(40)
# N_data_samples.append(40)
# N_kernel_samples.append(2000)
# batch_size.append(20)
# kernel_type.append('Gaussian')
# cost_func.append('Stein')
# qc.append('3q-qvm')
# score.append('Spectral') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)


'''################################'''
'''FOUR QUBITS'''
'''################################'''

# N_epochs.append(200)
# learning_rate.append(0.08)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('4q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.05)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('4q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.05)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.13)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('4q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.05)
# runs.append(0)


# N_epochs.append(200)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('4q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.1)
# runs.append(0)

# N_epochs.append(125)
# learning_rate.append(0.08)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(30)
# N_data_samples.append(30)
# N_kernel_samples.append(2000)
# batch_size.append(20)
# kernel_type.append('Gaussian')
# cost_func.append('Stein')
# qc.append('4q-qvm')
# score.append('Spectral') 
# stein_eigvecs.append(6)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)


# CompareCostFunctions(N_epochs, learning_rate, data_type, data_circuit,
#                         N_born_samples, N_data_samples, N_kernel_samples,
#                         batch_size, kernel_type, cost_func, qc, score,
#                         stein_eigvecs, stein_eta, sinkhorn_eps, runs, 'cost',  legend =True)

###################################################################################################################
# #Compute MMD Averages and error bars over certain number of runs
###################################################################################################################
def AverageCost(N_epochs, learning_rate, data_type, data_circuit,
                            N_born_samples, N_data_samples, N_kernel_samples,
                            batch_size, kernel_type, cost_func, qc, score,
                            stein_eigvecs, stein_eta, sinkhorn_eps, runs):
    '''
    This function reads in a number of runs, each run has the same parameters: computes average losses, and error and prints to a new file
    '''
    loss, born_final_probs, data_probs_final = ReadFromFile(N_epochs, learning_rate, data_type, data_circuit,
															N_born_samples, N_data_samples, N_kernel_samples,
															batch_size, kernel_type, cost_func, qc, score,
															stein_eigvecs, stein_eta, sinkhorn_eps, runs)
 
    N_runs = len(runs)
    TV_loss_total = np.zeros_like(loss[0]['TV']) 
    CostTrain_loss_total = np.zeros_like(loss[0][('%s' %cost_func[0], 'Train')]) 
    CostTest_loss_total = np.zeros_like(loss[0][('%s' %cost_func[0], 'Test')]) 
    

    [average_loss, error_upper, error_lower] = [{} for _ in range(3)]

    for run in range(N_runs):
        TV_loss_total           += loss[run]['TV']
        CostTrain_loss_total    += loss[run][('%s' %cost_func[run], 'Train')]
        CostTest_loss_total     += loss[run][('%s' %cost_func[run], 'Test')]
     
    N_epochs = N_epochs[0]

    average_loss['TV']                              = TV_loss_total/N_runs
    average_loss[('%s' %cost_func[run], 'Train')]   = CostTrain_loss_total/N_runs
    average_loss[('%s' %cost_func[run], 'Test')]    = CostTest_loss_total/N_runs
  
    
    [TV_max, TV_min, cost_train_max, cost_train_min, cost_test_max, cost_test_min] = [np.zeros(N_epochs-1) for _ in range(6)]
    
    for epoch in range(N_epochs-1):
        temp_tv  = []
        temp_cost_test = []    
        temp_cost_train  = []
        for run in range(N_runs):
            temp_tv.append(loss[run]['TV'][epoch])
            temp_cost_test.append(loss[run][('%s' %cost_func[run], 'Test')][epoch])
            temp_cost_train.append(loss[run][('%s' %cost_func[run], 'Train')][epoch])
        TV_max[epoch]          = max(temp_tv)
        cost_train_max[epoch]  = max(temp_cost_train)
        cost_test_max[epoch]   = max(temp_cost_test)
        TV_min[epoch]          = min(temp_tv)
        cost_train_min[epoch]  = min(temp_cost_train)
        cost_test_min[epoch]   = min(temp_cost_test)

    error_upper['TV']                           = np.absolute(average_loss['TV'] - TV_max)
    error_upper[('%s' %cost_func[0], 'Train')]  = np.absolute(average_loss[('%s' %cost_func[0], 'Train')]   - cost_train_max)
    error_upper[('%s' %cost_func[0], 'Test')]   = np.absolute(average_loss[('%s' %cost_func[0], 'Test')]    - cost_test_max)

    error_lower['TV']                           = np.absolute(average_loss['TV'] - TV_min)
    error_lower[('%s' %cost_func[0], 'Train')]  = np.absolute(average_loss[('%s' %cost_func[0], 'Train')]   - cost_train_min)
    error_lower[('%s' %cost_func[0], 'Test')]   = np.absolute(average_loss[('%s' %cost_func[0], 'Test')]    - cost_test_min)
   
    return  average_loss, error_upper, error_lower


def PrintAveragesToFiles(N_epochs, learning_rate, data_type, data_circuit,
                            N_born_samples, N_data_samples, N_kernel_samples,
                            batch_size, kernel_type, cost_func, qc, score,
                            stein_eigvecs, stein_eta, sinkhorn_eps, runs):

    if all(x == learning_rate[0] for x in learning_rate) is False:
        raise ValueError('All Learning Rates must be the same in all inputs.')
    elif all(x == sinkhorn_eps[0] for x in sinkhorn_eps) is False:
        raise ValueError('All Sinkhorn regularisers must be the same in all inputs.')
    average_loss, error_upper, error_lower = AverageCost(N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples,
                                                        batch_size, kernel_type, cost_func, qc, score,
                                                        stein_eigvecs, stein_eta, sinkhorn_eps, runs)

    stein_params = [score[0], stein_eigvecs[0], stein_eta[0], kernel_type[0]]
    N_samples =  [N_data_samples[0], N_born_samples[0], batch_size[0], N_kernel_samples[0]]
    trial_name = MakeTrialNameFile(cost_func[0], data_type[0], data_circuit[0], N_epochs[0],learning_rate[0], qc[0], kernel_type[0], N_samples, stein_params, sinkhorn_eps[0], 'Average')


    loss_path               = '%s/loss/%s/' %(trial_name, cost_func[0])
    TV_path                 = '%s/loss/TV/' %trial_name
    loss_path_upper_error   = '%s/loss/%s/upper_error/' %(trial_name, cost_func[0])
    loss_path_lower_error   = '%s/loss/%s/lower_error/' %(trial_name, cost_func[0])


    #create directories to store output training information
    MakeDirectory(loss_path)
    MakeDirectory(TV_path)

    MakeDirectory(loss_path_upper_error)
    MakeDirectory(loss_path_lower_error)

    #Print Upper Bounds on loss errors
    np.savetxt('%s/loss/%s/upper_error/train' 	%(trial_name,cost_func[0]),  	error_upper[('%s' %cost_func[0], 'Train')])
    np.savetxt('%s/loss/%s/upper_error/test' 	%(trial_name,cost_func[0]), 	error_upper[('%s' %cost_func[0], 'Test')] )

    np.savetxt('%s/loss/TV/upper_error' %(trial_name),  error_upper[('TV')]) 

    #Print Lower Bounds on loss errors
    np.savetxt('%s/loss/%s/lower_error/train' 	%(trial_name,cost_func[0]),  	error_lower[('%s' %cost_func[0], 'Train')])
    np.savetxt('%s/loss/%s/lower_error/test' 	%(trial_name,cost_func[0]), 	error_lower[('%s' %cost_func[0], 'Test')] )

    np.savetxt('%s/loss/TV/lower_error' %(trial_name),  error_lower[('TV')]) 

    np.savetxt('%s/loss/%s/train_avg' 	%(trial_name,cost_func[0]),  	average_loss[('%s' %cost_func[0], 'Train')])
    np.savetxt('%s/loss/%s/test_avg' 	%(trial_name,cost_func[0]), 	average_loss[('%s' %cost_func[0], 'Test')] )

    np.savetxt('%s/loss/TV/average' %(trial_name),  average_loss[('TV')]) #Print Total Variation of Distributions during training

    return 

# N_epochs.append(100)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('Aspen-4-4Q-A-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.1)
# runs.append(0)

# N_epochs.append(100)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('Aspen-4-4Q-A-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(0.1)
# runs.append(1)

# N_epochs.append(100)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('Aspen-4-4Q-A-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.1)
# runs.append(2)

# N_epochs.append(100)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('Aspen-4-4Q-A-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(0.1)
# runs.append(3)

# N_epochs.append(100)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('Aspen-4-4Q-A-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(0.1)
# runs.append(4)

# PrintAveragesToFiles(N_epochs, learning_rate, data_type, data_circuit,
#                             N_born_samples, N_data_samples, N_kernel_samples,
#                             batch_size, kernel_type, cost_func, qc, score,
#                             stein_eigvecs, stein_eta, sinkhorn_eps, runs)
####################################################################################################################
# #Plot Single Cost
###################################################################################################################

def PlotSingleCostFunction(N_epochs, learning_rate, data_type, data_circuit,
                            N_born_samples, N_data_samples, N_kernel_samples,
                            batch_size, kernel_type, cost_func, qc, score,
                            stein_eigvecs, stein_eta, sinkhorn_eps, comparison, legend = True):

    loss, born_final_probs, data_probs_final = ReadFromFile(N_epochs, learning_rate, data_type, data_circuit,
															N_born_samples, N_data_samples, N_kernel_samples,
															batch_size, kernel_type, cost_func, qc, score,
                                                            stein_eigvecs, stein_eta, sinkhorn_eps,0)
    x = np.arange(0, N_epochs-1, 1)

    if comparison.lower() == 'cost':												
        if legend == True:
            """WITH LEGEND"""
            try:
                average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs, learning_rate, data_type, data_circuit,	
                                                                            N_born_samples, N_data_samples, N_kernel_samples,
                                                                            batch_size, kernel_type, cost_func, qc, score,
                                                                            stein_eigvecs, stein_eta, sinkhorn_eps)

            except:
                pass
        
            if cost_func.lower() == 'mmd':
            
                try:
                    train_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                except:
                    pass
                plot_colour  = 'r'
                #For a single run
                # plt.plot(loss[('MMD', 'Train')],  '%so-' %(plot_colour), label =r'MMD, %i Train Points,  %i Born Samples with $\kappa_%s$.' \
                #                             %(round(N_data_samples*0.8), N_born_samples, kernel_type[0]))
                # plt.plot(loss[('MMD', 'Test')],  '%sx--' %(plot_colour), label =r'MMD, %i Test Points,  %i Born Samples with $\kappa_%s$.' \
                #                             %(N_data_samples - round(N_data_samples*0.8), N_born_samples, kernel_type[0]))
               
                try:
                    plt.errorbar(x, average_loss[('MMD', 'Train')], train_error, None,\
                                            '%sx-' %(plot_colour), label =r'%s, %i Train Points for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(cost_func,  round(N_data_samples*0.8), kernel_type[0], learning_rate),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                    plt.errorbar(x, average_loss[('MMD', 'Test')], test_error, None,\
                                            '%s-' %(plot_colour), label =r'%s, %i Test Points for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(cost_func, N_data_samples - round(N_data_samples*0.8), kernel_type[0], learning_rate),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                except:
                    plt.plot(loss[('MMD', 'Train')],  '%so-' %(plot_colour), label =r' %i Train Points using Exact Score wit' \
                                            %(round(N_data_samples*0.8)))
                    plt.plot(loss[('MMD', 'Test')],  '%sx--' %(plot_colour), label =r', %i Test Points using Exact Score with.' \
                                            %(N_data_samples - round(N_data_samples*0.8)))
                plt.ylabel(r'MMD Loss $\mathcal{L}_{\mathsf{MMD}}$', fontsize = 20)

            elif cost_func.lower() == 'sinkhorn':

                try:
                    train_error = np.vstack((lower_error[('Sinkhorn', 'Train')], upper_error[('Sinkhorn', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Sinkhorn', 'Train')], upper_error[('Sinkhorn', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                except:
                    pass
                plot_colour  = 'b'
                plt.errorbar(x, average_loss[('Sinkhorn', 'Train')], train_error, None,\
                                            '%sx-' %(plot_colour),  label =r'%s, %i Train Points using Hamming Cost' %(cost_func, N_data_samples - round(N_data_samples*0.8))\
                                                +'\n'+\
                                                r'$\eta_{init}$ = %.3f.' %learning_rate,\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                plt.errorbar(x, average_loss[('Sinkhorn', 'Test')], test_error, None,\
                                            '%s-' %(plot_colour), label =r'%s, %i Test Points using Hamming Cost' %(cost_func, N_data_samples - round(N_data_samples*0.8))\
                                                +'\n'+\
                                                r'$\eta_{init}$ = %.3f.' %learning_rate,\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                # plt.plot(loss[('Sinkhorn', 'Train')],  '%so-' %(plot_colour), label =r'Sinkhorn, %i Train Points with $\epsilon = %.3f $.' \
                #                             %(round(N_data_samples*0.8), sinkhorn_eps))
                # plt.plot(loss[('Sinkhorn', 'Test')],  '%sx--' %(plot_colour), label =r'Sinkhorn, %i Test Points with $\epsilon = %.3f$.' \
                #                             %(N_data_samples - round(N_data_samples*0.8), sinkhorn_eps))

                plt.ylabel(r'Sinkhorn Loss $\mathcal{L}_{\mathsf{SH}}$', fontsize = 20)
            elif cost_func.lower() == 'stein':

                try:
                    train_error = np.vstack((lower_error[('Stein', 'Train')], upper_error[('Stein', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Stein', 'Train')], upper_error[('Stein', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                except:
                    pass

                if score.lower() == 'exact':
                    plot_colour  = 'c'

                    plt.plot(loss[('Stein', 'Train')],  '%so-' %(plot_colour), label =r'Sinkhorn, %i Train Points using Exact Score' \
                                                %(round(N_data_samples*0.8)))
                    plt.plot(loss[('Stein', 'Test')],  '%sx--' %(plot_colour), label =r'Sinkhorn, %i Test Points using Exact Score ' \
                                                %(N_data_samples - round(N_data_samples*0.8)))
                elif score.lower() == 'spectral':
                    plot_colour  = 'm'
                    
                    plt.plot(loss[('Stein', 'Train')],  '%sx-' %(plot_colour), label =r'Sinkhorn, %i Train Points using Spectral Score' \
                                                %(round(N_data_samples*0.8)))
                    plt.plot(loss[('Stein', 'Test')],  '%s-' %(plot_colour), label =r'Sinkhorn, %i Test Points using Spectral Score.' \
                                                %(N_data_samples - round(N_data_samples*0.8)))

                plt.ylabel(r'Stein Loss $\mathcal{L}_{\mathsf{SD}}$', fontsize = 20)
            
            plt.xlabel("Epochs", fontsize = 20)
            plt.legend(loc='best', prop={'size': 20}).draggable()

            plt.show()

        elif legend == False:
            if cost_func.lower() == 'mmd':
                plot_colour  = 'r'

                plt.plot(loss[('MMD', 'Train')],  '%so-' %plot_colour)
                plt.plot(loss[('MMD', 'Test')],  '%sx--' %plot_colour)

                plt.ylabel(r'MMD Loss $\mathcal{L}^\theta_{\mathsf{MMD}}$')
                plt.title("MMD")

            elif cost_func.lower() == 'sinkhorn':

                plot_colour  = 'b'

                plt.plot(loss[('Sinkhorn', 'Train')],  '%so-' %plot_colour)
                plt.plot(loss[('Sinkhorn', 'Test')],  '%sx--' %plot_colour)

                plt.ylabel(r'Sinkhorn Loss $\mathcal{L}^\theta_{\mathsf{SH}}$')
                plt.title("Sinkhorn Divergence" )

            elif cost_func.lower() == 'stein':
                plot_colour  = 'g'

                plt.plot(loss[('Stein', 'Train')],  '%so-' %(plot_colour))
                plt.plot(loss[('Stein', 'Test')],  '%sx--' %(plot_colour))
                plt.ylabel(r'Sinkhorn Loss $\mathcal{L}_{\mathsf{SD}}$')
                plt.title("Stein Discrepancy")

            plt.xlabel("Epochs")

            plt.show()
    elif comparison.lower() == 'tv':
        try:
            average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs, learning_rate, data_type, data_circuit,	
                                                                            N_born_samples, N_data_samples, N_kernel_samples,
                                                                            batch_size, kernel_type, cost_func, qc, score,
                                                                            stein_eigvecs, stein_eta, sinkhorn_eps)
        except:
            pass
        if legend == True:
            """WITH LEGEND"""
            if cost_func.lower() == 'mmd':
                #Compute Average losses and errors, over a certain number of runs
                
                plot_colour  = 'r'
                #For a single run
                # plt.plot(loss[('MMD', 'Train')],  '%so-' %(plot_colour), label =r'MMD, %i Train Points,  %i Born Samples with $\kappa_%s$.' \
                #                             %(round(N_data_samples*0.8), N_born_samples, kernel_type[0]))
                # plt.plot(loss[('MMD', 'Test')],  '%sx--' %(plot_colour), label =r'MMD, %i Test Points,  %i Born Samples with $\kappa_%s$.' \
                #                             %(N_data_samples - round(N_data_samples*0.8), N_born_samples, kernel_type[0]))
                #For an averaged run
                x = np.arange(0, N_epochs-1, 1)
                try:
                    train_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                except:
                    pass
                if legend == True:
                    """WITH LEGEND"""
                    try:
                        x_mmd = np.arange(0, len(average_loss['TV']))
                        plt.errorbar(x_mmd, average_loss[('TV')], train_error, None,\
                                                '%sx-' %(plot_colour), label =r'%s, %i Train Points for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                    %(cost_func,  round(N_data_samples*0.8), kernel_type[0], learning_rate),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                        plt.errorbar(x_mmd, average_loss[('TV')], test_error, None,\
                                                '%s-' %(plot_colour), label =r'%s, %i Test Points for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                    %(cost_func, N_data_samples - round(N_data_samples*0.8), kernel_type[0], learning_rate),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    except:
                        plt.plot(loss[('TV')],  '%so-' %(plot_colour), label =r' %i Train Points using Exact Score with' \
                                                %(round(N_data_samples*0.8)))
                        plt.plot(loss[('TV')],  '%sx--' %(plot_colour), label =r', %i Test Points using Exact Score with.' \
                                                %(N_data_samples - round(N_data_samples*0.8)))
                plt.ylabel(r'MMD Loss $\mathcal{L}_{\mathsf{MMD}}$', fontsize = 20)

            elif cost_func.lower() == 'sinkhorn':
                try:
                    train_error = np.vstack((lower_error[('Sinkhorn', 'Train')], upper_error[('Sinkhorn', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Sinkhorn', 'Train')], upper_error[('Sinkhorn', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                except:
                    pass
                plot_colour  = 'b'
                # try:
                x_sink = np.arange(0, len(average_loss['TV']))

                plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Train')], train_error, None,\
                                            '%sx-' %(plot_colour), label =r'%s, %i Train Points for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(cost_func,  round(N_data_samples*0.8), kernel_type[0], learning_rate),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Test')], test_error, None,\
                                            '%s-' %(plot_colour), label =r'%s, %i Test Points for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(cost_func, N_data_samples - round(N_data_samples*0.8), kernel_type[0], learning_rate),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                # except:
                #     plt.plot(loss[('Sinkhorn', 'Train')],  '%sx-' %(plot_colour), label =r'Sinkhorn, %i Train Points with $\epsilon = %.3f $.' \
                #                                 %(round(N_data_samples*0.8), sinkhorn_eps))
                #     plt.plot(loss[('Sinkhorn', 'Test')],  '%s-' %(plot_colour), label =r'Sinkhorn, %i Test Points with $\epsilon = %.3f$.' \
                #                                 %(N_data_samples - round(N_data_samples*0.8), sinkhorn_eps))

                plt.ylabel(r'Sinkhorn Loss $\mathcal{L}_{\mathsf{SH}}$', fontsize = 20)
            elif cost_func.lower() == 'stein':
                try:
                    train_error = np.vstack((lower_error[('Stein', 'Train')], upper_error[('Stein', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Stein', 'Train')], upper_error[('Stein', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                except:
                    pass
                if score.lower() == 'exact':
                    plot_colour  = 'c'

                    plt.plot(loss['TV'],  '%so-' %(plot_colour), label =r'Stein, %i Train Points using Exact Score.' \
                                                %(round(N_data_samples*0.8)))
                elif score.lower() == 'spectral':
                    plot_colour  = 'm'
                    
                    plt.plot(loss['TV'],  '%so-' %(plot_colour), label =r'Stein, %i Train Points using Spectral Score.' \
                                                %(round(N_data_samples*0.8)))

                plt.ylabel(r'$\mathsf{TV}$', fontsize = 20)
            
            plt.xlabel("Epochs", fontsize = 20)
            plt.legend(loc='best', prop={'size': 20}).draggable()
                        

            plt.show()

        elif legend == False:
            if cost_func.lower() == 'mmd':
                plot_colour  = 'r'

                plt.plot(loss[('MMD', 'Train')],  '%so-' %plot_colour)
                plt.plot(loss[('MMD', 'Test')],  '%sx--' %plot_colour)

                plt.ylabel(r'MMD Loss $\mathcal{L}^\theta_{\mathsf{MMD}}$')
                plt.title("MMD")

            elif cost_func.lower() == 'sinkhorn':

                plot_colour  = 'b'

                plt.plot(loss[('Sinkhorn', 'Train')],  '%so-' %plot_colour)
                plt.plot(loss[('Sinkhorn', 'Test')],  '%sx--' %plot_colour)

                plt.ylabel(r'Sinkhorn Loss $\mathcal{L}^\theta_{\mathsf{SH}}$')
                plt.title("Sinkhorn Divergence" )

            elif cost_func.lower() == 'stein':
                plot_colour  = 'g'

                plt.plot(loss[('Stein', 'Train')],  '%so-' %(plot_colour))
                plt.plot(loss[('Stein', 'Test')],  '%sx--' %(plot_colour))
                plt.ylabel(r'Sinkhorn Loss $\mathcal{L}_{\mathsf{SD}}$')
                plt.title("Stein Discrepancy")

            plt.xlabel("Epochs")

            plt.show()
    elif comparison.lower() == 'probs':

        fig, axs = plt.subplots()
        # data_plot_colour = 'k'

        # bar_plot_colour = ['g', 'b', 'r', 'm', 'b', 'm-', 'k-', 'b+']
        data_plot_colour = 'r'

        bar_plot_colour = 'b'

        axs.clear()
        #Plot Data
        x = np.arange(len(data_probs_final))
        axs.bar(x, data_probs_final.values(), width=0.1, color= '%s' %data_plot_colour, align='center')
        #Plot one distribution
        axs.bar(x-(0.2*(0+0.5)), born_final_probs.values(), width=0.1, color='%s' %(bar_plot_colour), align='center')

        #Plot other costs
        # print(list(born_final_probs[-2].keys()), x)
        # axs.bar(x-(0.2*(0+1)), born_final_probs[-3].values(), width=0.1, color='%s' %(bar_plot_colour[1]), align='center')
        # axs.bar(x-(0.2*(0+1.5)), born_final_probs[-2].values(), width=0.1, color='%s' %(bar_plot_colour[2]), align='center')
        # axs.bar(x-(0.2*(0+2)), born_final_probs[-1].values(), width=0.1, color='%s' %(bar_plot_colour[3]), align='center')

        axs.set_xlabel("Outcomes", fontsize=20)
        axs.set_ylabel("Probability", fontsize=20)
        if legend == True:
            # axs.set_title(r'Outcome Distributions')
            axs.legend(('Data',r'\textsf{MMD}', r'Sinkhorn', r'Exact Stein',  r'Spectral Stein' ), fontsize = 20)

        axs.set_xticks(range(len(data_probs_final)))
        axs.set_xticklabels(list(data_probs_final.keys()),rotation=70)
    
        plt.show()
        return

'''3 QUBIT SINKHORN'''
# N_epochs            = 200
# learning_rate       = 0.1
# data_type           = 'Bernoulli_Data'
# data_circuit        ='IQP'
# N_born_samples      = 500
# N_data_samples      = 500
# N_kernel_samples    = 2000
# batch_size          = 250
# kernel_type         = 'Gaussian'
# cost_func           = 'Sinkhorn'
# qc                  = '3q-qvm'
# score               = 'Approx'
# stein_eigvecs       = 3                 
# stein_eta           = 0.01
# sinkhorn_eps        = 0.08
# runs                = 0

''''4 QUBIT SINKHORN'''
# N_epochs            = 200
# learning_rate       = 0.1
# data_type           = 'Bernoulli_Data'
# data_circuit        ='IQP'
# N_born_samples      = 500
# N_data_samples      = 500
# N_kernel_samples    = 2000
# batch_size          = 250
# kernel_type         = 'Gaussian'
# cost_func           = 'Sinkhorn'
# qc                  = '4q-qvm'
# score               = 'Approx'
# stein_eigvecs       = 3                 
# stein_eta           = 0.01
# sinkhorn_eps        = 0.1
# runs                = 0

''''4 QUBIT STEIN'''
# N_epochs            = 125
# learning_rate       = 0.08
# data_type           = 'Bernoulli_Data'
# data_circuit        ='IQP'
# N_born_samples      = 30
# N_data_samples      = 30
# N_kernel_samples    = 2000
# batch_size          = 20
# kernel_type         = 'Gaussian'
# cost_func           = 'Stein'
# qc                  = '4q-qvm'
# score               = 'Spectral'
# stein_eigvecs       = 6                 
# stein_eta           = 0.01
# sinkhorn_eps        = 0.08
# runs                = 0


# PlotSingleCostFunction(N_epochs, learning_rate, data_type, data_circuit,
#                             N_born_samples, N_data_samples, N_kernel_samples,
#                             batch_size, kernel_type, cost_func, qc, score,
#                             stein_eigvecs, stein_eta, sinkhorn_eps, 'cost',  legend = True)


# ###################################################################################################################
# #Compare Kernels
###################################################################################################################

        
def CompareKernelsPlot(N_epochs, learning_rate, data_type, data_circuit,
                        N_born_samples, N_data_samples, N_kernel_samples,
                        batch_size, kernel_type, cost_func, qc, score,
                        stein_eigvecs, stein_eta, sinkhorn_eps, comparison, runs, legend = True):
    loss, born_final_probs, data_probs_final = ReadFromFile(N_epochs, learning_rate, data_type, data_circuit,
                                                                N_born_samples, N_data_samples, N_kernel_samples,
                                                                batch_size, kernel_type, cost_func, qc, score,
                                                                stein_eigvecs, stein_eta, sinkhorn_eps, runs)
    # print(len(N_epochs), loss)
    if all(x.lower() == 'mmd' for x in cost_func) is False:
        #If all cost functions to be compared are the mmd, 
        raise  ValueError('All cost functions must be MMD')
    else:
        if comparison.lower() == 'tv':
            plot_colour = ['rx-', 'r+-', 'ro-', 'bx-', 'b+-', 'bo-']
        elif comparison.lower() == 'mmd':
            if qc[0][0].lower() == '2':
                plot_colour = ['rx-', 'r+-', 'ro-', 'bx-', 'b+-', 'bo-']
            elif qc[0][0].lower() == '4':
                plot_colour = ['rx-', 'bx-']
        
    N_trials = len(N_epochs)
    x = np.arange(0, N_epochs[0]-1, 1)
    if comparison.lower() == 'probs':

        fig, axs = plt.subplots()

        axs.clear()
        x = np.arange(len(data_probs_final[0]))
        axs.bar(x, data_probs_final[0].values(), width=0.2, color= 'g' , align='center')
        axs.bar(x-(0.2*(0+1)), born_final_probs[2].values(), width=0.2, color='b', align='center')
        axs.bar(x-(0.2*(0+2)), born_final_probs[-1].values(), width=0.2, color='r', align='center')

        axs.set_xlabel("Outcomes", fontsize=20)
        axs.set_ylabel("Probability", fontsize=20)

        if legend == True:
            # axs.set_title(r'\textsf{IBM} and data distribution with $\kappa_G$ vs. $\kappa_Q$ for %i qubits' %(N_qubits))
            axs.legend(('Data',r'MMD with $\kappa_G$',r'MMD with $\kappa_Q$'), fontsize=20).draggable()

        axs.set_xticks(range(len(data_probs_final[0])))
        axs.set_xticklabels(list(data_probs_final[0].keys()),rotation=70)

        plt.show()
    else:
        for trial in range(N_trials): 
            #Compute Average losses and errors, over a certain number of runs
            average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs[trial], learning_rate[trial], data_type[trial], data_circuit[trial],	
                                                                        N_born_samples[trial], N_data_samples[trial], N_kernel_samples[trial],
                                                                        batch_size[trial], kernel_type[trial], cost_func[trial], qc[trial], score[trial],
                                                                        stein_eigvecs[trial], stein_eta[trial], sinkhorn_eps[trial]) 
            tv_error = np.vstack((lower_error['TV'], upper_error['TV'])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            mmd_train_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            mmd_test_error = np.vstack((lower_error[('MMD', 'Test')], upper_error[('MMD', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function

            if comparison.lower() == 'tv':
                if cost_func[trial].lower() == 'mmd':
                
            
                    if legend == True:   
                        """WITH LEGENDS"""	
                        plt.errorbar(x, average_loss['TV'], tv_error, None,\
                                                    '%s' %(plot_colour[trial]), label =r'%s, for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                    %(cost_func[trial],kernel_type[trial][0], learning_rate[trial]),\
                                                    capsize=0, elinewidth=0.5, markeredgewidth=2)
                        plt.legend(loc='best', prop={'size': 15}).draggable()


                    elif legend == False:   
                        """WITHOUT LEGENDS"""
                        plt.errorbar(x, average_loss['TV'], tv_error, None,'%s' %(plot_colour[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)	

                plt.ylabel("TV", fontsize=20)
        
            elif comparison.lower() == 'mmd':
                if legend == True:  
                    if qc[trial].lower()[0] == '2':
                        plot_markers = ['x', '+', 'o', 'x', '+', 'o'] 
                    elif qc[trial].lower()[0] == '4':
                        plot_markers = ['x', 'x', 'x', 'x', 'x', 'x'] 
                    """WITH LEGENDS"""		
                    # plt.errorbar(x, average_loss[('MMD', 'Train')], mmd_train_error, None, '%s' %(plot_colour[trial]), label =r'MMD, %i Train Points, with $\kappa_%s$, $\eta_{init}$ = %.3f.' \
                    #             %(round(N_data_samples[trial]*0.8), kernel_type[trial][0], learning_rate[trial]), 
                    #             capsize=0, elinewidth=0.5, markeredgewidth=2)
                    if kernel_type[trial][0].lower() == 'q': 
                        plt.errorbar(x, average_loss[('MMD', 'Train')], mmd_train_error, None, 'r%s-' %plot_markers[trial], label =r'MMD, %i Train Points for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                    %(round(N_data_samples[trial]*0.8), kernel_type[trial][0], learning_rate[trial]), 
                                    capsize=0, elinewidth=0.5, markeredgewidth=2)
                        plt.errorbar(x, average_loss[('MMD', 'Test')], mmd_train_error, None, 'r-', label =r'MMD, %i Test Points' \
                                    %(N_data_samples[trial] - round(N_data_samples[trial]*0.8)), 
                                    capsize=0, elinewidth=0.5, markeredgewidth=2)
                    elif kernel_type[trial][0].lower() == 'g': 
                        plt.errorbar(x, average_loss[('MMD', 'Train')], mmd_train_error, None, 'b%s-' %plot_markers[trial], label =r'MMD, %i Train Points for $\kappa_{%s}$, $\eta_{init}$ = %.3f.'\
                                    %(round(N_data_samples[trial]*0.8) ,kernel_type[trial][0], learning_rate[trial] ), 
                                    capsize=0, elinewidth=0.5, markeredgewidth=2) 
                        plt.errorbar(x, average_loss[('MMD', 'Test')], mmd_train_error, None, 'b-', label =r'MMD, %i Test Points.' \
                                    %(N_data_samples[trial] - round(N_data_samples[trial]*0.8)), 
                                    capsize=0, elinewidth=0.5, markeredgewidth=2) 
                   
                    plt.legend(loc='best', prop={'size': 20}).draggable()

                    # plt.title(r'MMD for %i qubits between \textsf{IBM} and data with $\kappa_G$ vs. $\kappa_Q$' % N_qubits)

                elif legend == False:   
                    """WITHOUT LEGENDS"""
                    plt.errorbar(x, average_loss[('MMD', 'Train')], mmd_train_error, None,'%s' %(plot_colour[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)

                    if kernel_type[trial][0].lower() == 'q': 
                        plt.errorbar(x, average_loss[('MMD', 'Test')], mmd_test_error, None,'r-',\
                                                        capsize=0, elinewidth=0.5, markeredgewidth=2)	

                    elif kernel_type[trial][0].lower() == 'g':
                        plt.errorbar(x, average_loss[('MMD', 'Test')], mmd_test_error, None,'b-',\
                                                        capsize=0, elinewidth=1, markeredgewidth=2)	
                plt.ylabel("MMD", fontsize=20)

            plt.xlabel("Epochs", fontsize=20)
        plt.show()

    
    return

[N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples, batch_size, kernel_type, \
    cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, runs] = [[] for _ in range(16)] 


# N_epochs.append(200)
# learning_rate.append(0.01)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Quantum')
# cost_func.append('MMD')
# qc.append('2q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.05)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Quantum')
# cost_func.append('MMD')
# qc.append('2q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.08)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Quantum')
# cost_func.append('MMD')
# qc.append('2q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.05)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Quantum')
# cost_func.append('MMD')
# qc.append('4q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.05)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.01)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('2q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.05)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.05)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('4q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.08)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('4q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('4q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_qubits = int(qc[0][0])

# CompareKernelsPlot(N_epochs, learning_rate, data_type, data_circuit,
#                         N_born_samples, N_data_samples, N_kernel_samples,
#                         batch_size, kernel_type, cost_func, qc, score,
#                         stein_eigvecs, stein_eta, sinkhorn_eps, 'mmd', runs,  legend = True)


# ###################################################################################################################
# #Automatic Compilation
# ###################################################################################################################



def PlotAutomaticCompilation(N_epochs, learning_rate, data_type, data_circuit,
							N_born_samples, N_data_samples, N_kernel_samples,
							batch_size, kernel_type1, cost_func, qc, score,
							stein_eigvecs1, stein_eta, sinkhorn_eps, runs, comparison, legend = True):
    '''This function reads output information from a file, relating to automatric compilation of circuits, and plots'''
    loss, born_final_probs, data_probs_final = ReadFromFile(N_epochs, learning_rate, data_type, data_circuit,
                                                                N_born_samples, N_data_samples, N_kernel_samples,
                                                                batch_size, kernel_type1, cost_func, qc, score,
                                                                stein_eigvecs1, stein_eta, sinkhorn_eps, runs)
                                            
    try:
        average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs, learning_rate, data_type, data_circuit,	
                                                                                N_born_samples, N_data_samples, N_kernel_samples,
                                                                                batch_size, kernel_type, cost_func, qc, score,
                                                                                stein_eigvecs, stein_eta, sinkhorn_eps)
        cost_error = np.vstack((lower_error['TV'], upper_error['TV'])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
    except:
        print('Average files not found')
        pass

    x = np.arange(0, N_epochs-1, 1)

    if comparison.lower() == 'tv':
        if legend == True:
            """WITH LEGEND"""
            if cost_func.lower() == 'mmd':
                plot_colour = 'g'
                try:
                    plt.errorbar(x, average_loss['TV'], cost_error, None,\
                                            '%s' %(plot_colour), label =r'MMD, %i Data Samples for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(N_data_samples, kernel_type[0], learning_rate),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                except:
                    plt.plot(loss[('TV')],  '%so-' %(plot_colour[0]), label =r'MMD, %i Data Samples, $\eta_{init}$ = %.3f.' \
                                        %( N_data_samples,learning_rate))
         
            elif cost_func.lower() == 'sinkhorn':
                plot_colour = 'b'

                try:
                    plt.errorbar(x, average_loss['TV'], cost_error, None,\
                                                '%s' %(plot_colour), label =r'Sinkhorn, %i Data Samples with Hamming Cost' %(N_data_samples)+ \
                                                '\n'\
                                                + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                except:
                    print('Average File Not Found')
                    plt.plot(loss[('TV')],  '%so-' %(plot_colour[0]), label =r'Sinkhorn, %i Data Samples with Hamming Cost' %(N_data_samples)+ \
                                                '\n'\
                                                + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %(learning_rate, sinkhorn_eps))
         
            elif cost_func.lower() == 'stein':
                plot_colour = 'r'

                try:
                    plt.errorbar(x, average_loss['TV'], cost_error, None,\
                                            '%s' %(plot_colour), label =r'Stein, %i Data Samples, using %s Score,$\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(N_data_samples, score, kernel_type[0], learning_rate),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                except:
                    plt.plot(loss[('TV')],  '%so-' %(plot_colour[0]), label =r'Stein, %i Data Samples, using %s Score, $\kappa_{%s}$, $\eta_{init}$ = %.3f. ' \
                                        %( N_data_samples, score, kernel_type[0], learning_rate))


            plt.xlabel("Epochs", fontsize  =20)
            plt.ylabel("TV", fontsize  =20)
            plt.legend(loc='best', prop={'size': 15}).draggable()
                    
        elif legend == False:
            """WITHOUT LEGEND"""							
            plot_colour = ['r', 'b', 'g']
            plt.plot(loss[('TV')],  '%so-' %plot_colour[0])

        else: raise ValueError('*legend* must be either *True* or *False*')
        plt.show()

    elif comparison.lower() == 'cost':
        try: 
            if cost_func.lower() == 'mmd':
                    train_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('MMD', 'Test')], upper_error[('MMD', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            elif cost_func.lower() == 'stein':
                    train_error = np.vstack((lower_error[('Stein', 'Train')], upper_error[('Stein', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Stein', 'Test')], upper_error[('Stein', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            elif cost_func.lower() == 'sinkhorn':
                    train_error = np.vstack((lower_error[('Sinkhorn', 'Train')], upper_error[('Sinkhorn', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Sinkhorn', 'Test')], upper_error[('Sinkhorn', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
        except:
                pass

        if legend == True:
            """WITH LEGEND"""
            if cost_func.lower() == 'mmd':
                plot_colour = 'g'
                try:
                    x_mmd = np.arange(0, len(average_loss['MMD', 'Train']))
                    plt.errorbar(x_mmd, average_loss[('MMD', 'Train')], train_error, None,\
                                                '%so' %(plot_colour), label =r'MMD, %i Train Points using $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(round(N_data_samples*0.8), kernel_type[0], learning_rate),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                    plt.errorbar(x_mmd, average_loss[('MMD', 'Test')], test_error, None,\
                                                '%s-' %(plot_colour), label =r'MMD, %i Test Points using $\kappa_{%s}$, $\eta_{init}$ = %.3f.' \
                                                %(N_data_samples - round(N_data_samples*0.8),kernel_type[0], learning_rate),\
                                                capsize=1, elinewidth=1, markeredgewidth=2)
                except:
                    plt.plot(loss[('MMD', 'Train')],  '%so-' %(plot_colour[0]), label =r'%s, %i Train Points,   $\eta_{init}$ = %.3f.' \
                                                %(cost_func, N_data_samples, learning_rate))
                    plt.plot(loss[('MMD', 'Test')],  '%so-' %(plot_colour[0]), label =r'%s, %i Test Points,  $\eta_{init}$ = %.3f.' \
                                                %(cost_func, N_data_samples, learning_rate))
                plt.ylabel(r'$\mathsf{MMD}$ Loss $\mathcal{L}_{\mathsf{MMD}}$', fontsize  =20)
            elif cost_func.lower() == 'sinkhorn':
                plot_colour = 'b'
                try:
                    x_sink = np.arange(0, len(average_loss['Sinkhorn', 'Train']))

                    plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Train')], train_error, None,\
                                                    '%so' %(plot_colour), label =r'Sinkhorn, %i Train Samples with Hamming Cost' %(round(N_data_samples*0.8))+ \
                                                    '\n'\
                                                    + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Test')], test_error, None,\
                                                    '%s-' %(plot_colour), label =r'Sinkhorn, %i Test Samples with Hamming Cost' %(N_data_samples - round(N_data_samples*0.8))+ \
                                                    '\n'\
                                                    + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps),\
                                                   capsize=1, elinewidth=1, markeredgewidth=2)
                 
                except:
                    plt.plot(loss[('Sinkhorn', 'Train')],  '%so-' %(plot_colour), label =r'Sinkhorn, %i Train Samples with Hamming Cost' %(round(N_data_samples*0.8))+ \
                                                    '\n'\
                                                    + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps))
                    plt.plot(loss[('Sinkhorn', 'Test')],  '%s-' %(plot_colour),label =r'Sinkhorn, %i Test Samples with Hamming Cost' %(N_data_samples - round(N_data_samples*0.8))+ \
                                                    '\n'\
                                                    + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps))
                plt.ylabel(r'$\mathsf{Sinkhorn}$ Loss $\mathcal{L}_{\mathsf{SH}}$', fontsize  =20)

            elif cost_func.lower() == 'stein':
                plot_colour = 'r'
                try:
                    x_sink = np.arange(0, len(average_loss['Stein', 'Train']))

                    plt.errorbar(x_sink, average_loss[('Stein', 'Train')], train_error, None,\
                                                    '%so' %(plot_colour), label =r'Stein, %i Train Samples with  $\kappa_{%s}$' %(round(N_data_samples*0.8),kernel_type[0])+ \
                                                    '\n'\
                                                    + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    plt.errorbar(x_sink, average_loss[('Stein', 'Test')], test_error, None,\
                                                    '%s-' %(plot_colour), label =r'Stein, %i Test Samples with  $\kappa_{%s}$' %(N_data_samples - round(N_data_samples*0.8),kernel_type[0])+ \
                                                    '\n'\
                                                    + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                except:
                    plt.plot(loss[('Stein', 'Train')],  '%so-' %(plot_colour), label =r'Stein, %i Train Samples with  $\kappa_{%s}$' %(round(N_data_samples*0.8),kernel_type[0])+ \
                                                    '\n'\
                                                    + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps))
                    plt.plot(loss[('Stein', 'Test')],  '%so-' %(plot_colour), label =r'Stein, %i Test Samples with  $\kappa_{%s}$' %(N_data_samples - round(N_data_samples*0.8),kernel_type[0])+ \
                                                    '\n'\
                                                    + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps))
                plt.ylabel(r'$\mathsf{Stein}$ Loss $\mathcal{L}_{\mathsf{SD}}$', fontsize  = 20)
  
            plt.xlabel("Epochs", fontsize  =20)
            plt.legend(loc='best', prop={'size': 20}).draggable()
                    

        elif legend == False:
            """WITHOUT LEGEND"""							
            plot_colour = ['r', 'b', 'g']
            plt.plot(loss[('TV')],  '%so-' %plot_colour[0])

        else: raise ValueError('*legend* must be either *True* or *False*')
       
        plt.show()
    elif comparison.lower() == 'probs':

        fig, axs = plt.subplots()

        axs.clear()
        x = np.arange(len(data_probs_final))
        axs.bar(x, data_probs_final.values(), width=0.2, color= 'k' , align='center')
        axs.bar(x-(0.2*(0+1)), born_final_probs.values(), width=0.2, color='b', align='center')

        axs.set_xlabel("Outcomes", fontsize=20)
        axs.set_ylabel("Probability", fontsize=20)

        if legend == True:
            # axs.set_title(r'\textsf{IBM} and data distribution with $\kappa_G$ vs. $\kappa_Q$ for %i qubits' %(N_qubits))
            axs.legend((r'IQP Data',r'$\mathsf{QAOA}$ $\mathsf{IBM}$ with Sinkhorn.'), fontsize = 20)

        axs.set_xticks(range(len(data_probs_final)))
        axs.set_xticklabels(list(data_probs_final.keys()),rotation=70)

        plt.show()
    return

'''TWO QUBITS'''

# N_epochs = 100
# learning_rate =  0.05
# data_type = 'Quantum_Data'
# data_circuit = 'IQP'
# N_born_samples = 1000
# N_data_samples = 1000
# N_kernel_samples = 2000
# batch_size = 250
# kernel_type ='Gaussian'
# cost_func = 'Sinkhorn'
# qc = '2q-qvm'
# score = 'Approx' 
# stein_eigvecs = 3                 
# stein_eta = 0.01  
# sinkhorn_eps = 0.1
# runs = 0

'''THREE QUBITS'''

# N_epochs = 125
# learning_rate =  0.05
# data_type = 'Quantum_Data'
# data_circuit = 'IQP'
# N_born_samples = 500
# N_data_samples = 500
# N_kernel_samples = 2000
# batch_size = 250
# kernel_type ='Gaussian'
# cost_func = 'Sinkhorn'
# qc = '3q-qvm'
# score = 'Approx' 
# stein_eigvecs = 3                 
# stein_eta = 0.01  
# sinkhorn_eps = 0.1
# runs = 3

# PlotAutomaticCompilation(N_epochs, learning_rate, data_type, data_circuit,
# 							N_born_samples, N_data_samples, N_kernel_samples,
# 							batch_size, kernel_type, cost_func, qc, score,
# 							stein_eigvecs, stein_eta, sinkhorn_eps, runs, 'cost', legend = True)


def CompareCostFunctionsonQPU(N_epochs, learning_rate, data_type, data_circuit,
                        N_born_samples, N_data_samples, N_kernel_samples,
                        batch_size, kernel_type, cost_func, qc, score,
                        stein_eigvecs, stein_eta, sinkhorn_eps, runs, comparison, legend = True):
    loss, born_final_probs, data_probs_final = ReadFromFile(N_epochs, learning_rate, data_type, data_circuit,
															N_born_samples, N_data_samples, N_kernel_samples,
															batch_size, kernel_type, cost_func, qc, score,
															stein_eigvecs, stein_eta, sinkhorn_eps, runs)

    
    plot_colour = ['g-', 'g*-', 'b-', 'b*-']
    N_trials = len(N_epochs)
    if comparison.lower() == 'probs':

        fig, axs = plt.subplots()
        data_plot_colour = 'k'


        bar_plot_colour = ['g','b']
        patterns = [ "o" , ""]

        axs.clear()
        #Plot Data
        x = np.arange(len(data_probs_final[0]))
        axs.bar(x, data_probs_final[0].values(), width=0.1, color= '%s' %data_plot_colour, align='center')
        #Plot MMD One
        axs.bar(x-(0.2*(0+1)), born_final_probs[1].values(), width=0.1, color='%s' %(bar_plot_colour[0]), hatch=patterns[0],  align='center')
        axs.bar(x-(0.2*(0+0.5)), born_final_probs[0].values(), width=0.1, color='%s' %(bar_plot_colour[0]), hatch=patterns[1], align='center')

        #Plot Sinkhorn
        axs.bar(x-(0.2*(0+2)),      born_final_probs[3].values(), width=0.1, color='%s' %(bar_plot_colour[1]),    hatch=patterns[0],  align='center')
        axs.bar(x-(0.2*(0+1.5)),    born_final_probs[2].values(), width=0.1, color='%s' %(bar_plot_colour[1]),    hatch=patterns[1],  align='center')

        axs.set_xlabel("Outcomes", fontsize=20)
        axs.set_ylabel("Probability", fontsize=20)
        if legend == True:
            # axs.set_title(r'Outcome Distributions')
            axs.legend(('Data',r'\textsf{MMD}, on %s' %qc[0], r'\textsf{MMD}, on %s' %qc[1], r'Sinkhorn, on %s' %qc[2], r'Sinkhorn, on %s' %qc[3] ), fontsize = 20).draggable()

        axs.set_xticks(range(len(data_probs_final[0])))
        axs.set_xticklabels(list(data_probs_final[0].keys()),rotation=70)
    
        plt.show()
    elif comparison.lower() == 'tv':
        for trial in range(N_trials):
            #Compute Average losses and errors, over a certain number of runs
            try:
                average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs[trial], learning_rate[trial], data_type[trial], data_circuit[trial],	
                                                                                    N_born_samples[trial], N_data_samples[trial], N_kernel_samples[trial],
                                                                                    batch_size[trial], kernel_type[trial], cost_func[trial], qc[trial], score[trial],
                                                                                    stein_eigvecs[trial], stein_eta[trial], sinkhorn_eps[trial])
                cost_error = np.vstack((lower_error['TV'], upper_error['TV'])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            except:
                pass
                # raise FileNotFoundError('The Average cost could not be found')

            x = np.arange(0, N_epochs[trial]-1, 1)

            if legend == True:
                """WITH LEGEND"""
                ##If the trial is running perfectly on the QVM
                if cost_func[trial].lower() == 'mmd':
                    # try:
                    plt.errorbar(x, average_loss['TV'], cost_error, None,\
                                                '%s' %(plot_colour[trial]), label =r'MMD, on %s, $\eta_{init}$ = %.3f.' \
                                                    %( qc[trial], learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    # except:
                    #     plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]), label =r'MMD, on %s  $\eta_{init}$ = %.3f.' \
                    #                                     %(qc[trial], learning_rate[trial]))
                elif  cost_func[trial].lower() == 'stein':

                    x_stein = np.arange(0, len(average_loss['TV']))
                    if score[trial].lower() == 'exact':
                        try:
                        
                            plt.errorbar(x_stein, average_loss['TV'], cost_error, None,\
                                                        '%s' %(plot_colour[trial]), label =r'Stein, on %s  %i Data Samples using Exact Score $\eta_{init}$ = %.3f.' \
                                                        %(qc[trial], N_data_samples[trial], learning_rate[trial]),\
                                                            capsize=1, elinewidth=1, markeredgewidth=2)
                    
                        except:
                            plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]), label =r'Stein, on %s %i Data Samples using Exact Score  $\eta_{init}$ = %.3f.' \
                                                        %(qc[trial], N_data_samples[trial],learning_rate[trial]))
                    elif score[trial].lower() == 'spectral':
                        try:
                            plt.errorbar(x_stein, average_loss['TV'], cost_error, None,\
                                                        '%s' %(plot_colour[trial]), label =r'Stein on %s using Spectral Score$\eta_{init}$ = %.3f.' \
                                                        %(qc[trial], learning_rate[trial]),\
                                                            capsize=1, elinewidth=1, markeredgewidth=2)
                    
                        except:

                            plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]), label =r'Stein, on %s using Spectral Score $\eta_{init}$ = %.3f.' \
                                                            %(qc[trial], N_data_samples[trial],  learning_rate[trial]))
                elif cost_func[trial].lower() == 'sinkhorn':

                    try:
                        x_sink = np.arange(0, len(average_loss['TV']))

                        plt.errorbar(x_sink, average_loss['TV'], cost_error, None,\
                                                    '%s' %(plot_colour[trial]), label =r'Sinkhorn, on %s,  $\eta_{init}$ = %.3f.' \
                                                        %(qc[trial], learning_rate[trial]),\
                                                        capsize=1, elinewidth=1, markeredgewidth=2)
                    except:
                        plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]), label =r'Sinkhorn, on %s, $\eta_{init}$ = %.3f.' \
                                                        %(qc[trial],  learning_rate[trial]))
            
            elif legend == False:
                """WITHOUT LEGEND"""
                plt.plot(loss[trial][('TV')],  '%s' %(plot_colour[trial]))
                
        plt.xlabel("Epochs", fontsize=20)
        plt.ylabel("TV", fontsize=20)


        plt.legend(loc='best', prop={'size': 20}).draggable()
        plt.show()       


    elif comparison.lower() == 'cost':
        for trial in range(N_trials):
            try:
                
                average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs[trial], learning_rate[trial], data_type[trial], data_circuit[trial],	
                                                                                    N_born_samples[trial], N_data_samples[trial], N_kernel_samples[trial],
                                                                                    batch_size[trial], kernel_type[trial], cost_func[trial], qc[trial], score[trial],
                                                                                    stein_eigvecs[trial], stein_eta[trial], sinkhorn_eps[trial])
                if cost_func[trial].lower() == 'mmd':
                    train_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('MMD', 'Test')], upper_error[('MMD', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                elif cost_func[trial].lower() == 'stein':
                    train_error = np.vstack((lower_error[('Stein', 'Train')], upper_error[('Stein', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Stein', 'Test')], upper_error[('Stein', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                elif cost_func[trial].lower() == 'sinkhorn':
                    train_error = np.vstack((lower_error[('Sinkhorn', 'Train')], upper_error[('Sinkhorn', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Sinkhorn', 'Test')], upper_error[('Sinkhorn', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            except:
                pass
            if cost_func[trial].lower() == 'mmd':
                if qc[trial].lower()[8] == '3':
                    if learning_rate[trial] == 0.1:
                        plot_colour = ['c', 'g']
                    elif learning_rate[trial] == 0.15:
                        plot_colour = ['c', 'r']
                    else: print('Plot colours have not been assigned for this LR choice')
                elif qc[trial].lower()[8] == '4':
                        plot_colour = ['c', 'g']

                try:

                    x_mmd = np.arange(0, len(average_loss['MMD', 'Train']))

                    plt.errorbar(x_mmd, average_loss[('MMD', 'Train')], train_error, None,\
                                                    '%sx-' %(plot_colour[trial]), label =r'MMD on %s, %i Train Samples,$\eta_{init}$ = %.3f.' \
                                                    %(qc[trial], round(N_data_samples[trial]*0.8), learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    plt.errorbar(x_mmd, average_loss[('MMD', 'Test')], test_error, None,\
                                                    '%s-' %(plot_colour[trial]), label =r'MMD on %s, %i Test Samples,  $\eta_{init}$ = %.3f.' \
                                                    %(qc[trial], N_data_samples[trial] - round(N_data_samples[trial]*0.8), learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                except:
                    plt.plot(loss[trial][('MMD', 'Train')],  '%so-' %(plot_colour[trial]), label =r'MMD on %s, %i Train Samples, $\eta_{init}$ = %.3f.' \
                                                    %(qc[trial], round(N_data_samples[trial]*0.8), kernel_type[trial][0], learning_rate[trial]))
                    plt.plot(loss[trial][('MMD', 'Test')],  '%s-' %(plot_colour[trial]), label =r'MMD on %s, %i Test Samples, $\eta_{init}$ = %.3f.' \
                                                    %(qc[trial],  N_data_samples[trial] - round(N_data_samples[trial]*0.8), learning_rate[trial]))
    
                plt.ylabel(r'MMD Loss $\mathcal{L}_{\mathsf{MMD}}$', fontsize = 20)
            
            elif cost_func[trial].lower() == 'stein':
                if score[trial].lower() == 'exact':
                    plot_colour  = 'r'
                    try:
                        x_stein = np.arange(0, len(average_loss['Stein', 'Train']))

                        plt.errorbar(x_stein, average_loss[('Stein', 'Train')], train_error, None,\
                                                    '%so' %(plot_colour), label =r'Stein, on %s, %i Train Points using Exact Score, $\eta_{init}$ = %.3f.' \
                                                    %(qc[trial], round(N_data_samples[trial]*0.8),  learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                        plt.errorbar(x_stein, average_loss[('Stein', 'Test')], test_error, None,\
                                                    '%s-' %(plot_colour), label =r'Stein, on %s, %i Test Points using Exact Score, $\eta_{init}$ = %.3f.' \
                                                    %(qc[trial], N_data_samples[trial] - round(N_data_samples[trial]*0.8), learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    except:
                        plt.plot(loss[trial][('Stein', 'Train')],  '%so-' %(plot_colour), label =r'Stein, on %s, %i Train Points using Exact Score, $\eta_{init}$ = %.3f.' \
                                                    %(qc[trial],  round(N_data_samples[trial]*0.8), learning_rate[trial]))
                        plt.plot(loss[trial][('Stein', 'Test')],  '%s--' %(plot_colour), label =r'Stein, on %s, %i Test Points  using Exact Score, $\eta_{init}$ = %.3f.' \
                                                    %(qc[trial], N_data_samples[trial] - round(N_data_samples[trial]*0.8), learning_rate[trial]))
                elif score[trial].lower() == 'spectral':
                    plot_colour  = 'm'
                    
                    plt.plot(loss[trial][('Stein', 'Train')],  '%so-' %(plot_colour), label =r'Stein, on %s, %i Train Points  using Spectral Score, $\eta_{init}$ = %.3f.' \
                                                %(qc[trial], round(N_data_samples[trial]*0.8), learning_rate[trial]))

                    plt.plot(loss[trial][('Stein', 'Test')],  '%s-' %(plot_colour), label =r'Stein, on %s, %i Test Points using Spectral Score, $\eta_{init}$ = %.3f.' \
                                                %(qc[trial], N_data_samples[trial] - round(N_data_samples[trial]*0.8),  learning_rate[trial]))

                plt.ylabel(r'Stein Loss $\mathcal{L}_{\mathsf{SD}}$', fontsize = 20)
            elif cost_func[trial].lower() == 'sinkhorn': 
                print(qc[trial].lower(), learning_rate[trial])        
                if qc[trial].lower()[8] == '3':
                    if learning_rate[trial] == 0.08:
                        plot_colour = ['c', 'b']
                    else: print('Plot colours have not been assigned for this LR choice')
                elif qc[trial].lower()[8] == '4':
                    if learning_rate[trial] == 0.1:
                        plot_colour = ['c', 'b']
                    else: print('Plot colours have not been assigned for this LR choice')

                try:
                    x_sink = np.arange(0, len(average_loss['Sinkhorn', 'Train']))

                    plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Train')], train_error, None,\
                                                    '%sx-' %(plot_colour[trial]), label =r'Sinkhorn, on %s, %i Train Samples, $\eta_{init}$ = %.3f.' \
                                                    %(qc[trial], round(N_data_samples[trial]*0.8),  learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                    plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Test')], test_error, None,\
                                                    '%s-' %(plot_colour[trial]), label =r'Sinkhorn, on %s, %i Test Samples, $\eta_{init}$ = %.3f.' \
                                                    %(qc[trial], N_data_samples[trial] - round(N_data_samples[trial]*0.8), learning_rate[trial]),\
                                                    capsize=1, elinewidth=1, markeredgewidth=2)
                except:
                    plt.plot(loss[trial][('Sinkhorn', 'Train')],  '%so-' %(plot_colour), label =r'Sinkhorn, on %s, %i Train Samples, $\eta_{init}$ = %.3f.' \
                                                %(qc[trial], round(N_data_samples[trial]*0.8),  learning_rate[trial]))
                    plt.plot(loss[trial][('Sinkhorn', 'Test')],  '%s--' %(plot_colour), label =r'Sinkhorn, on %s, %i Test Samples, $\eta_{init}$ = %.3f.' \
                                                %(qc[trial], N_data_samples[trial] - round(N_data_samples[trial]*0.8), learning_rate[trial]))
    
                plt.ylabel(r'Sinkhorn Loss $\mathcal{L}_{\mathsf{SH}}$', fontsize = 20)
        plt.xlabel("Epochs", fontsize = 20)
        plt.legend(loc='best', prop={'size': 20}).draggable()

        plt.show()
    return

[N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples, batch_size, kernel_type, \
cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, runs] = [[] for _ in range(16)] 


'''#################################'''
'''ON CHIP ASPEN-4-3Q-A'''
'''#################################'''

# N_epochs.append(100)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('Aspen-4-3Q-A')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(100)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('Aspen-4-3Q-A-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(100)
# learning_rate.append(0.08)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('Aspen-4-3Q-A')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.1)
# runs.append(0)

# N_epochs.append(100)
# learning_rate.append(0.08)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('Aspen-4-3Q-A-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.1)
# runs.append(0)


'''#################################'''
'''ON CHIP ASPEN-4-4Q-A'''
'''#################################'''

# N_epochs.append(100)
# learning_rate.append(0.15)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('Aspen-4-4Q-A')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(100)
# learning_rate.append(0.15)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('MMD')
# qc.append('Aspen-4-4Q-A-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(100)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('Aspen-4-4Q-A')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.1)
# runs.append(0)

# N_epochs.append(100)
# learning_rate.append(0.1)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('Aspen-4-4Q-A-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.1)
# # runs.append(0)

# CompareCostFunctionsonQPU(N_epochs, learning_rate, data_type, data_circuit,
#                         N_born_samples, N_data_samples, N_kernel_samples,
#                         batch_size, kernel_type, cost_func, qc, score,
#                         stein_eigvecs, stein_eta, sinkhorn_eps, runs, 'cost',  legend =True)