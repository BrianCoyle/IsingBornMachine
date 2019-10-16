import numpy as np
import ast
import sys
import json
from auxiliary_functions import SampleListToArray
import matplotlib
from matplotlib import rc

rc('text', usetex=True)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})

matplotlib.rc('xtick', labelsize=30)     
matplotlib.rc('ytick', labelsize=30) 
import matplotlib.pyplot as plt
from file_operations_in import ReadFromFile, AverageCostsFromFile
from file_operations_out import MakeTrialNameFile, MakeDirectory

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

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
        plot_colour = ['green', 'darkorange', 'c', 'blue', 'red', 'm']
  
    N_trials = len(N_epochs)
    if comparison.lower() == 'probs':

        fig, axs = plt.subplots()
        data_plot_colour = 'k'

        axs.clear()
        x = np.arange(len(data_probs_final[0]))

        bar_plot_colour = ['green', 'blue', 'red', 'm']

            #Plot MMD
        axs.bar(x, data_probs_final[0].values(), width=0.1, color= '%s' %data_plot_colour, align='center')

        axs.bar(x-(0.2*(0+0.5)), born_final_probs[-5].values(), width=0.1, color='%s' %(bar_plot_colour[-4]), align='center')

        axs.bar(x-(0.2*(0+1)), born_final_probs[-3].values(), width=0.1, color='%s' %(bar_plot_colour[-3]), align='center')
        axs.bar(x-(0.2*(0+1.5)), born_final_probs[-2].values(), width=0.1, color='%s' %(bar_plot_colour[-2]), align='center')
        axs.bar(x-(0.2*(0+2)), born_final_probs[-1].values(), width=0.1, color='%s' %(bar_plot_colour[-1]), align='center')
        
        axs.legend(('Data',r'\textsf{MMD}', r'Sinkhorn', r'Exact Stein',  r'Spectral Stein' ), fontsize = 20)
      
        axs.set_xticks(range(len(data_probs_final[0])))
        axs.set_xticklabels(list(data_probs_final[0].keys()),rotation=70)

      
    elif comparison.lower() == 'tv':
        fig, ax = plt.subplots()
        if qc[0][0].lower() == '3':
            axins = zoomed_inset_axes(ax, 5, loc='center') 
            x1, x2, y1, y2 = 190, 200, 0.00, 0.021 # specify the limits
        elif qc[0][0].lower() == '4':
            axins = zoomed_inset_axes(ax, 2.5, loc='center right') 
            x1, x2, y1, y2 = 180, 200, 0.02, 0.06 # specify the limits
        elif qc[0][0].lower() == '5':
            axins = zoomed_inset_axes(ax, 2.5, loc='upper left') 
            x1, x2, y1, y2 = 0,1 , 0.24, 0.25 # specify the limits

        axins.set_xlim(x1, x2) # apply the x-limits
        axins.set_ylim(y1, y2) # apply the y-limits
        plt.xticks(visible=False)
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")

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

            x = np.arange(0, N_epochs[trial]-1, 1)

            if cost_func[trial].lower() == 'mmd':
        
                ax.plot(x, average_loss['TV'], '-', color ='%s' % plot_colour[trial] , label =r'$\mathsf{MMD}$ for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' %( kernel_type[trial][0], learning_rate[trial]))
                ax.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor= plot_colour[trial] )

                axins.plot(x, average_loss['TV'], color ='%s' % plot_colour[trial] , label =r'$\mathsf{MMD}$  for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' %( kernel_type[trial][0], learning_rate[trial]))
                axins.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor= plot_colour[trial] )

            elif cost_func[trial].lower() == 'stein':

                if score[trial].lower() == 'exact':
                    # plot_colour  = 'r'

                    ax.plot(x, average_loss['TV'], '-', color ='%s' % plot_colour[trial] , label =r'Stein using Exact score for $\eta_{init}$ = %.3f.'% learning_rate[trial])
                    ax.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor= plot_colour[trial] )
                    axins.plot(x, average_loss['TV'], '-', color ='%s' % plot_colour[trial] , label =r'Stein using Exact score for $\eta_{init}$ = %.3f.'% learning_rate[trial])
                    axins.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor= plot_colour[trial] )
                   
                elif score[trial].lower() == 'spectral':
                    
                    # plot_colour  = 'm'

                    ax.plot(loss[trial][('TV')], '-', color ='%s' % plot_colour[trial], label =r'Stein using Spectral score for $\eta_{init}$ = %.3f.' \
                                                    % learning_rate[trial])
                    axins.plot(loss[trial][('TV')],   color ='%s' % plot_colour[trial] , label =r'Stein using Spectral score for $\eta_{init}$ = %.3f.' \
                                                    %learning_rate[trial])
            elif cost_func[trial].lower() == 'sinkhorn':
                
                ax.plot(x, average_loss['TV'],'-', color ='%s' % plot_colour[trial] , label =r'Sinkhorn using Hamming cost, $\eta_{init}$ = %.3f.' % learning_rate[trial] )
                ax.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor= plot_colour[trial] )

                axins.plot(x, average_loss['TV'], '-', color ='%s' % plot_colour[trial]  , label =r'Sinkhorn using Hamming cost, $\eta_{init}$ = %.3f.' % learning_rate[trial] )
                axins.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor= plot_colour[trial]  )

        ax.legend(loc='best', prop={'size': 20})

    elif comparison.lower() == 'cost':
        for trial in range(N_trials):
            try:
                average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs[trial], learning_rate[trial], data_type[trial], data_circuit[trial],	
                                                                                    N_born_samples[trial], N_data_samples[trial], N_kernel_samples[trial],
                                                                                    batch_size[trial], kernel_type[trial], cost_func[trial], qc[trial], score[trial],
                                                                                    stein_eigvecs[trial], stein_eta[trial], sinkhorn_eps[trial])
            except:
                print('Average Not found')
                loss, born_final_probs, data_probs_final = ReadFromFile(N_epochs[trial], learning_rate[trial], data_type[trial], data_circuit[trial],	
                                                                                    N_born_samples[trial], N_data_samples[trial], N_kernel_samples[trial],
                                                                                    batch_size[trial], kernel_type[trial], cost_func[trial], qc[trial], score[trial],
                                                                                    stein_eigvecs[trial], stein_eta[trial], sinkhorn_eps[trial], runs[trial])
            if cost_func[trial].lower() == 'mmd':
                plot_colour  = ['c', 'y', 'g']
                x = np.arange(0, len(average_loss['MMD', 'Train']))

        
                plt.plot(x, average_loss['MMD', 'Train'],'%so-' % plot_colour[trial],\
                                label =r'$\mathsf{MMD}$ on training set using $\eta_{init}$ = %.3f.' % learning_rate[trial] )
                plt.fill_between(x, average_loss['MMD', 'Train'] - lower_error['MMD', 'Train'],\
                                        average_loss['MMD', 'Train'] + upper_error['MMD', 'Train'], facecolor= plot_colour[trial], alpha=0.3)

                plt.plot(x, average_loss['MMD', 'Test'],'%s-' % plot_colour[trial],\
                            label =r'$\mathsf{MMD}$ on test set using $\eta_{init}$ = %.3f.' % learning_rate[trial] )
                plt.fill_between(x, average_loss['MMD', 'Test'] - lower_error['MMD', 'Test'],\
                                        average_loss['MMD', 'Test'] + upper_error['MMD', 'Test'], alpha=0.3, facecolor= plot_colour[trial], interpolate=True)
                
            elif cost_func[trial].lower() == 'stein':
                if score[trial].lower() == 'exact':
                    plot_colour  = 'r'
                    x = np.arange(0, len(average_loss['Stein', 'Train']))
                    plt.plot(x, average_loss['Stein', 'Train'],'%so-' % plot_colour,\
                                    label =r'Stein using Exact score, $\eta_{init}$ = %.3f.' % learning_rate[trial] )
                    plt.fill_between(x, average_loss['Stein', 'Train'] - lower_error['Stein', 'Train'],\
                                            average_loss['Stein', 'Train'] + upper_error['Stein', 'Train'], alpha=0.3, facecolor=plot_colour)

                elif score[trial].lower() == 'spectral':
                    plot_colour  = 'm'
                    
                    plt.plot(loss[('Stein', 'Train')],  '%so-' % plot_colour, \
                            label =r'Stein on training set using Spectral score, $\eta_{init}$ = %.3f.' %(learning_rate[trial] ))

                    plt.plot(loss[('Stein', 'Test')], '%s-' % plot_colour,\
                            label =r'Stein on test set using Spectral score, $\eta_{init}$ = %.3f.' %( learning_rate[trial]))

            elif cost_func[trial].lower() == 'sinkhorn':
                plot_colour  = 'b'
                x = np.arange(0, len(average_loss['Sinkhorn', 'Train']))

                plt.plot(x, average_loss['Sinkhorn', 'Train'],'%so-' % plot_colour,\
                                label =r'Sinkhorn on training set using Hamming cost, $\eta_{init}$ = %.3f.' % learning_rate[trial] )
                plt.fill_between(x, average_loss['Sinkhorn', 'Train'] - lower_error['Sinkhorn', 'Train'],\
                                        average_loss['Sinkhorn', 'Train'] + upper_error['Sinkhorn', 'Train'], alpha=0.3)

                plt.plot(x, average_loss['Sinkhorn', 'Test'],'%s-' % plot_colour,\
                            label =r'Sinkhorn on test set using Hamming cost, $\eta_{init}$ = %.3f.' % learning_rate[trial] )
                plt.fill_between(x, average_loss['Sinkhorn', 'Test'] - lower_error['Sinkhorn', 'Test'],\
                                        average_loss['Sinkhorn', 'Test'] + upper_error['Sinkhorn', 'Test'], alpha=0.3, facecolor=plot_colour, interpolate=True)
                
        plt.legend(loc='best', prop={'size': 20})
                
    plt.show()
    return

[N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples, batch_size, kernel_type, \
cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, runs] = [[] for _ in range(16)] 


'''THREE QUBITS'''

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
# qc.append('3q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)    
# sinkhorn_eps.append(0.01)
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
# qc.append('3q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)    
# sinkhorn_eps.append(1)
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
# sinkhorn_eps.append(1)
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
# cost_func.append('Sinkhorn')
# qc.append('3q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)    
# sinkhorn_eps.append(0.1)
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
# cost_func.append('Stein')
# qc.append('3q-qvm')
# score.append('Exact') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)



# N_epochs.append(200)
# learning_rate.append(0.01)
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
# stein_eigvecs.append(4)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)



'''################################'''
'''FOUR QUBITS'''
'''################################'''

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
# qc.append('4q-qvm')
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
# learning_rate.append(0.05)
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
# sinkhorn_eps.append(1)
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
# qc.append('4q-qvm')
# score.append('Exact') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.1)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.01)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(50)
# N_data_samples.append(50)
# N_kernel_samples.append(2000)
# batch_size.append(25)
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
#                         stein_eigvecs, stein_eta, sinkhorn_eps, runs, 'probs',  legend =True)

###################################################################################################################
# #Compute MMD Averages and error bars over certain number of runs
###################################################################################################################
def AverageCost(N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples,
                            batch_size, kernel_type, cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, runs):
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


def PrintAveragesToFiles(N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples,
                            batch_size, kernel_type, cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, runs):

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

# N_epochs.append(200)
# learning_rate.append(0.01)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('5q-qvm')
# score.append('Approx')  
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(1)
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
# cost_func.append('Sinkhorn')
# qc.append('5q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(1)
# runs.append(1)

# N_epochs.append(200)
# learning_rate.append(0.01)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('5q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(1)
# runs.append(2)

# N_epochs.append(200)
# learning_rate.append(0.01)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('5q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(1)
# runs.append(3)

# N_epochs.append(200)
# learning_rate.append(0.01)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Gaussian')
# cost_func.append('Sinkhorn')
# qc.append('5q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(1)
# runs.append(4)



# PrintAveragesToFiles(N_epochs, learning_rate, data_type, data_circuit,N_born_samples, N_data_samples, N_kernel_samples,
#                             batch_size, kernel_type, cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, runs)



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
        average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs, learning_rate, data_type, data_circuit,	N_born_samples, N_data_samples, N_kernel_samples,
                                                                    batch_size, kernel_type, cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps)

     
    
        if cost_func.lower() == 'mmd':
        
            try:
                train_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                test_error = np.vstack((lower_error[('MMD', 'Test')], upper_error[('MMD', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            except:
                pass
            plot_colour  = 'r'
         

            plt.plot(x, average_loss['MMD', 'Train'],'%so-' % plot_colour,\
                            label =r'MMD on training set for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' % (kernel_type[0], learning_rate) )
            plt.fill_between(x, average_loss['MMD', 'Train'] - lower_error['MMD', 'Train'],\
                                    average_loss['MMD', 'Train'] + upper_error['MMD', 'Train'], alpha=0.3, facecolor='%s'%plot_colour)

            plt.plot(x, average_loss['MMD', 'Test'],'%s-' % plot_colour,\
                            label =r'MMD on test set for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' % (kernel_type[0], learning_rate) )
            plt.fill_between(x, average_loss['MMD', 'Test'] - lower_error['MMD', 'Test'],\
                                    average_loss['MMD', 'Test'] + upper_error['MMD', 'Test'], alpha=0.3)

        elif cost_func.lower() == 'sinkhorn':

            try:
                train_error = np.vstack((lower_error[('Sinkhorn', 'Train')], upper_error[('Sinkhorn', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                test_error = np.vstack((lower_error[('Sinkhorn', 'Test')], upper_error[('Sinkhorn', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            except:
                pass

            plot_colour  = 'b'
            x = np.arange(0, len(average_loss['Sinkhorn', 'Train']))

            plt.plot(x, average_loss['Sinkhorn', 'Train'],'%so-' % plot_colour,\
                            label =r'Sinkhorn on training set, $\eta_{init}$ = %.3f.' % learning_rate )
            plt.fill_between(x, average_loss['Sinkhorn', 'Train'] - lower_error['Sinkhorn', 'Train'],\
                                    average_loss['Sinkhorn', 'Train'] + upper_error['Sinkhorn', 'Train'], alpha=0.5, facecolor=plot_colour)

            plt.plot(x, average_loss['Sinkhorn', 'Test'],'%s-' % plot_colour,\
                        label =r'Sinkhorn on test set, $\eta_{init}$ = %.3f.' % learning_rate )
            plt.fill_between(x, average_loss['Sinkhorn', 'Test'] - lower_error['Sinkhorn', 'Test'],\
                                    average_loss['Sinkhorn', 'Test'] + upper_error['Sinkhorn', 'Test'], alpha=0.3, facecolor=plot_colour)
            
         
        elif cost_func.lower() == 'stein':

            if score.lower() == 'exact':
                plot_colour  = 'c'

                plt.plot(loss[('Stein', 'Train')],  '%so-' %(plot_colour), label =r'Stein, on training set using Exact score' )
                plt.plot(loss[('Stein', 'Test')],  '%sx--' %(plot_colour), label =r'Stein, on test set using Exact score ' )
            elif score.lower() == 'spectral':
                plot_colour  = 'm'
                
                plt.plot(loss[('Stein', 'Train')],  '%sx-' %(plot_colour), label =r'Stein, on training set using Spectral score' )
                plt.plot(loss[('Stein', 'Test')],  '%s-' %(plot_colour), label =r'Stein, on test set using Spectral score.' )

        plt.legend(loc='best', prop={'size': 20})

        plt.show()

    elif comparison.lower() == 'tv':
        try:
            average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs, learning_rate, data_type, data_circuit,	N_born_samples, N_data_samples, N_kernel_samples,
                                                                            batch_size, kernel_type, cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps)
        except:
            pass
        
        x = np.arange(0, N_epochs-1, 1)

        if cost_func.lower() == 'mmd':
            #Compute Average losses and errors, over a certain number of runs
            
            plot_colour  = 'r'
                       
            plt.plot(x, loss['TV'], label =r'MMD on test set for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' % (kernel_type[0], learning_rate) )

            # plt.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2)
            
        elif cost_func.lower() == 'sinkhorn':
       
            plot_colour  = 'b'
            
            plt.plot(x, average_loss['TV'],label =r'Sinkhorn using Hamming cost, $\eta_{init}$ = %.3f.' % learning_rate )
            plt.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2)
            
     
        elif cost_func.lower() == 'stein':
          
            if score.lower() == 'exact':
                plot_colour  = 'c'

                plt.plot(loss['TV'],  '%so-' %(plot_colour), label =r'Stein using Exact score.')
            elif score.lower() == 'spectral':
                plot_colour  = 'm'
                
                plt.plot(loss['TV'],  '%so-' %(plot_colour), label =r'Stein using Spectral score.')
        
        plt.legend(loc='best', prop={'size': 20})
                    
        plt.show()

        return

'''3 QUBIT SINKHORN'''
# N_epochs            = 200
# learning_rate       = 0.01
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
# sinkhorn_eps        = 0.1
# runs                = 0

''''4 QUBIT SINKHORN'''
# N_epochs            = 200
# learning_rate       = 0.05
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
# sinkhorn_eps        = 1
# runs                = 0

# PlotSingleCostFunction(N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples,
#                             batch_size, kernel_type, cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, 'cost',  legend = True)


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
        #If all cost functions to be compared are the mmd
        raise  ValueError('All cost functions must be MMD')
    else:
        if comparison.lower() == 'tv':
            if qc[0][0].lower() == '2':
                plot_colour = ['rs-', 'r+-', 'ro-', 'bs-', 'b+-', 'bo-']
            elif qc[0][0].lower() == '3':
                plot_colour = ['rs-', 'b+-', 'ro-', 'bs-', 'b+-', 'bo-']
            elif qc[0][0].lower() == '4':
                plot_colour = ['rx-', 'bx-', 'c+-', 'mo-']
            elif qc[0][0].lower() == '5':
                plot_colour = ['rx-', 'bx-']
        elif comparison.lower() == 'mmd':
            if qc[0][0].lower() == '2':
                plot_colour = ['rs-', 'b+-', 'ro-', 'bs-', 'b+-', 'bo-']
            elif qc[0][0].lower() == '3':
                plot_colour = ['rs-', 'b+-', 'ro-', 'bs-', 'b+-', 'bo-']
            elif qc[0][0].lower() == '4':
                plot_colour = ['rx-', 'bx-']
        
    N_trials = len(N_epochs)
    x = np.arange(0, N_epochs[0]-1, 1)

    if comparison.lower() == 'probs':

        fig, axs = plt.subplots()

        axs.clear()
        x = np.arange(len(data_probs_final[0]))
        axs.bar(x, data_probs_final[0].values(), width=0.2, color= 'k' , align='center')
        axs.bar(x-(0.2*(0+1)), born_final_probs[2].values(), width=0.2, color='b', align='center')
        axs.bar(x-(0.2*(0+2)), born_final_probs[-1].values(), width=0.2, color='r', align='center')

        axs.legend(('Data',r'$\mathsf{MMD}$ with $\kappa_G$',r'$\mathsf{MMD}$ with $\kappa_Q$'), fontsize=20)

        axs.set_xticks(range(len(data_probs_final[0])))
        axs.set_xticklabels(list(data_probs_final[0].keys()),rotation=70)

        plt.show()

    else:
        fig, ax = plt.subplots()
        
        if comparison.lower() == 'tv':
            if qc[0][0].lower() == '2':
                plot_colour = ['rs-', 'b+-', 'ro-', 'bs-', 'b+-', 'bo-']
            elif qc[0][0].lower() == '3':
                axins = zoomed_inset_axes(ax, 2.5, loc='center') 
                x1, x2, y1, y2 = 190, 200, 0.01, 0.03 # specify the limits
            elif qc[0][0].lower() == '4':
                axins = zoomed_inset_axes(ax, 2.5, loc='upper center') 
                x1, x2, y1, y2 = 180, 200, 0.04, 0.09 # specify the limits
            elif qc[0][0].lower() == '5':
                axins = zoomed_inset_axes(ax, 2.5, loc='upper center') 
                x1, x2, y1, y2 = 180, 200, 0.03, 0.09 # specify the limits

            axins.set_xlim(x1, x2) # apply the x-limits
            axins.set_ylim(y1, y2) # apply the y-limits
            plt.xticks(visible=False)
            mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")
        if comparison.lower() == 'mmd':
            if qc[0][0].lower() == '2':
                plot_colour = ['rs-', 'b+-', 'ro-', 'bs-', 'b+-', 'bo-']
            elif qc[0][0].lower() == '3':
                axins = zoomed_inset_axes(ax, 2.5, loc='center') 
                x1, x2, y1, y2 = 180, 200, 0.00, 0.04 # specify the limits
            elif qc[0][0].lower() == '4':
                axins = zoomed_inset_axes(ax, 1.5, loc='center') 
                x1, x2, y1, y2 = 180, 200, 0.00, 0.04 # specify the limits
            elif qc[0][0].lower() == '5':
                axins = zoomed_inset_axes(ax, 2.5, loc='center') 
                x1, x2, y1, y2 = 180, 200, 0.03, 0.09 # specify the limits

            axins.set_xlim(x1, x2) # apply the x-limits
            axins.set_ylim(y1, y2) # apply the y-limits
            plt.xticks(visible=False)
            mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.5")

        for trial in range(N_trials): 
            #Compute Average losses and errors, over a certain number of runs
            average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs[trial], learning_rate[trial], data_type[trial], data_circuit[trial],	
                                                                        N_born_samples[trial], N_data_samples[trial], N_kernel_samples[trial],
                                                                        batch_size[trial], kernel_type[trial], cost_func[trial], qc[trial], score[trial],
                                                                        stein_eigvecs[trial], stein_eta[trial], sinkhorn_eps[trial]) 
            tv_error = np.vstack((lower_error['TV'], upper_error['TV'])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            mmd_train_error = np.vstack((lower_error[('MMD', 'Train')], upper_error[('MMD', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            mmd_test_error = np.vstack((lower_error[('MMD', 'Test')], upper_error[('MMD', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
            
            if qc[trial].lower()[0] == '2':
                    plot_markers = ['s', '+', 'o', 's', '+', 'o'] 
            elif qc[trial].lower()[0] == '3':
                    plot_markers = ['s', '+', 'o', 's', '+', 'o'] 
            elif qc[trial].lower()[0] == '4':
                    plot_markers = ['s', '+', 'o', 's', '+', 'o'] 
            elif qc[trial].lower()[0] == '5':
                    plot_markers = ['s', '+', 'o', 's', '+', 'o'] 
            
            if comparison.lower() == 'tv':

                if kernel_type[trial][0].lower() == 'q': 
                    # ax.plot(overview_data_x, overview_data_y)
                    ax.plot(x, average_loss['TV'], 'r%s-' %plot_markers[trial], \
                            label =r'$\mathsf{MMD}$ for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' %(kernel_type[trial][0], learning_rate[trial]) )

                    ax.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor='r')
                    axins.plot(x, average_loss['TV'], 'r%s-' %plot_markers[trial], \
                            label =r'$\mathsf{MMD}$ for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' %(kernel_type[trial][0], learning_rate[trial]) )

                    axins.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor='r')
                
                elif kernel_type[trial][0].lower() == 'g': 

                    ax.plot(x, average_loss['TV'], 'b%s-' %plot_markers[trial], \
                        label =r'$\mathsf{MMD}$ for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' %(kernel_type[trial][0], learning_rate[trial]) )
                    ax.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor='c')

                    axins.plot(x, average_loss['TV'], 'b%s-' %plot_markers[trial], \
                        label =r'$\mathsf{MMD}$ for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' %(kernel_type[trial][0], learning_rate[trial]) )
                    axins.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor='c')
                
                ax.legend(loc='best', prop={'size': 20})
        
            elif comparison.lower() == 'mmd':
            
                if kernel_type[trial][0].lower() == 'q': 
            
                    ax.plot(x, average_loss['MMD', 'Train'], 'r%s-' %plot_markers[trial], \
                                        label =r'$\mathsf{MMD}$ for $\kappa_Q$, $\eta_{init}$ = %.3f.'%learning_rate[trial])

                    ax.fill_between(x, average_loss['MMD', 'Train'] - lower_error['MMD', 'Train'],\
                                    average_loss['MMD', 'Train'] + upper_error['MMD', 'Train'], alpha=0.3, facecolor='r')

                    # plt.plot(x, average_loss['MMD', 'Test'], 'r-', label =r'MMD for $\kappa_Q$, $\eta_{init}$ = %.3f.'%learning_rate[trial])
                    ax.plot(x, average_loss['MMD', 'Test'], 'r-')
                    ax.fill_between(x, average_loss['MMD', 'Test'] - lower_error['MMD', 'Test'],\
                                    average_loss['MMD', 'Test'] + upper_error['MMD', 'Test'], alpha=0.1, facecolor='r')
                    axins.plot(x, average_loss['MMD', 'Train'], 'r%s-' %plot_markers[trial], \
                                        label =r'$\mathsf{MMD}$ for $\kappa_Q$, $\eta_{init}$ = %.3f.'%learning_rate[trial])

                    axins.fill_between(x, average_loss['MMD', 'Train'] - lower_error['MMD', 'Train'],\
                                    average_loss['MMD', 'Train'] + upper_error['MMD', 'Train'], alpha=0.3, facecolor='r')

                    # plt.plot(x, average_loss['MMD', 'Test'], 'r-', label =r'MMD for $\kappa_Q$, $\eta_{init}$ = %.3f.'%learning_rate[trial])
                    axins.plot(x, average_loss['MMD', 'Test'], 'r-')
                    axins.fill_between(x, average_loss['MMD', 'Test'] - lower_error['MMD', 'Test'],\
                                    average_loss['MMD', 'Test'] + upper_error['MMD', 'Test'], alpha=0.1, facecolor='r')

                elif kernel_type[trial][0].lower() == 'g': 
                    
                    ax.plot(x, average_loss['MMD', 'Train'], 'b%s-' %plot_markers[trial], \
                                        label =r'$\mathsf{MMD}$ for $\kappa_G$, $\eta_{init}$ = %.3f.'%learning_rate[trial])

                    ax.fill_between(x, average_loss['MMD', 'Train'] - lower_error['MMD', 'Train'],\
                                    average_loss['MMD', 'Train'] + upper_error['MMD', 'Train'], alpha=0.3, facecolor='b')

                    # plt.plot(x, average_loss['MMD', 'Test'], 'b-', label =r'$\mathsf{MMD}$ for $\kappa_G$, $\eta_{init}$ = %.3f.'%learning_rate[trial])
                    ax.plot(x, average_loss['MMD', 'Test'], 'b-')

                    ax.fill_between(x, average_loss['MMD', 'Test'] - lower_error['MMD', 'Test'],\
                                    average_loss['MMD', 'Test'] + upper_error['MMD', 'Test'], alpha=0.1, facecolor='b')
                    axins.plot(x, average_loss['MMD', 'Train'], 'b%s-' %plot_markers[trial], \
                                        label =r'$\mathsf{MMD}$ for $\kappa_G$, $\eta_{init}$ = %.3f.'%learning_rate[trial])

                    axins.fill_between(x, average_loss['MMD', 'Train'] - lower_error['MMD', 'Train'],\
                                    average_loss['MMD', 'Train'] + upper_error['MMD', 'Train'], alpha=0.3, facecolor='b')

                    # plt.plot(x, average_loss['MMD', 'Test'], 'b-', label =r'$\mathsf{MMD}$ for $\kappa_G$, $\eta_{init}$ = %.3f.'%learning_rate[trial])
                    axins.plot(x, average_loss['MMD', 'Test'], 'b-')

                    axins.fill_between(x, average_loss['MMD', 'Test'] - lower_error['MMD', 'Test'],\
                                    average_loss['MMD', 'Test'] + upper_error['MMD', 'Test'], alpha=0.1, facecolor='b')


                ax.legend(loc='best', prop={'size': 20})

        plt.show()

    
    return

[N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples, batch_size, kernel_type, \
    cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, runs] = [[] for _ in range(16)] 

'''
Three QUBITS
'''

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
# kernel_type.append('Quantum')
# cost_func.append('MMD')
# qc.append('3q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.075)
# data_type.append('Bernoulli_Data')
# data_circuit.append('IQP')
# N_born_samples.append(500)
# N_data_samples.append(500)
# N_kernel_samples.append(2000)
# batch_size.append(250)
# kernel_type.append('Quantum')
# cost_func.append('MMD')
# qc.append('3q-qvm')
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
# cost_func.append('MMD')
# qc.append('3q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                
# stein_eta.append(0.01)      
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.075)
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

'''
FOUR QUBITS
'''

# N_epochs.append(200)
# learning_rate.append(0.005)
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
# sinkhorn_eps.append(0.08)
# runs.append(0)


# N_epochs.append(200)
# learning_rate.append(0.007)
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
# sinkhorn_eps.append(0.08)
# runs.append(0)


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
# qc.append('4q-qvm')
# score.append('Approx') 
# stein_eigvecs.append(3)                 
# stein_eta.append(0.01) 
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(200)
# learning_rate.append(0.005)
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
# learning_rate.append(0.007)
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
# learning_rate.append(0.01)
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




# CompareKernelsPlot(N_epochs, learning_rate, data_type, data_circuit,
#                         N_born_samples, N_data_samples, N_kernel_samples,
#                         batch_size, kernel_type, cost_func, qc, score,
#                         stein_eigvecs, stein_eta, sinkhorn_eps, 'tv', runs,  legend = True)


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
     
        if cost_func.lower() == 'sinkhorn':
            plot_colour = 'b'

            # plt.errorbar(x, average_loss['TV'], cost_error, None,\
            #                                 '%s' %(plot_colour), label =r'Sinkhorn with Hamming cost'+ '\n'\
            #                                 + r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' %( learning_rate, sinkhorn_eps),\
            #                                 capsize=1, elinewidth=1, markeredgewidth=2)
            
            x_sink = np.arange(0, len(average_loss['TV']))

            plt.plot(x_sink, average_loss['TV'],label ='Sinkhorn using Hamming cost,'+'\n'+ r'$\eta_{init}$ = %.3f, $\epsilon$ = %.3f.' % (learning_rate, sinkhorn_eps) )
            plt.fill_between(x_sink, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.3)
            
            
            plt.legend(loc='best', prop={'size': 20})
                    
     
        plt.show()

    elif comparison.lower() == 'cost':
        try: 
            if cost_func.lower() == 'sinkhorn':
                    train_error = np.vstack((lower_error[('Sinkhorn', 'Train')], upper_error[('Sinkhorn', 'Train')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
                    test_error = np.vstack((lower_error[('Sinkhorn', 'Test')], upper_error[('Sinkhorn', 'Test')])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
        except:
                pass

        if cost_func.lower() == 'sinkhorn':
            plot_colour = 'b'
            x_sink = np.arange(0, len(average_loss['Sinkhorn', 'Train']))
            
            plt.plot(x_sink, average_loss['Sinkhorn', 'Train'],'%so-' % plot_colour,\
                            label =r'Sinkhorn on training set using Hamming cost, $\eta_{init}$ = %.3f.' % learning_rate )
            plt.fill_between(x_sink, average_loss['Sinkhorn', 'Train'] - lower_error['Sinkhorn', 'Train'],\
                                    average_loss['Sinkhorn', 'Train'] + upper_error['Sinkhorn', 'Train'], alpha=0.3, facecolor='b')

            plt.plot(x_sink, average_loss['Sinkhorn', 'Test'],'%s-' % plot_colour,\
                        label =r'Sinkhorn on test set using Hamming cost, $\eta_{init}$ = %.3f.' % learning_rate )
            plt.fill_between(x_sink, average_loss['Sinkhorn', 'Test'] - lower_error['Sinkhorn', 'Test'],\
                                    average_loss['Sinkhorn', 'Test'] + upper_error['Sinkhorn', 'Test'], alpha=0.3)
            
            # plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Train')], train_error, None,\
            #                                 '%so-' %(plot_colour), label =r'Sinkhorn on training set',\
            #                                 capsize=1, elinewidth=1, markeredgewidth=2)
            # plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Test')], test_error, None,\
            #                                 '%s-' %(plot_colour), label =r'Sinkhorn on test set',\
            #                                 capsize=1, elinewidth=1, markeredgewidth=2)
            
    
            plt.legend(loc='best', prop={'size': 20})
    
       
        plt.show()
    elif comparison.lower() == 'probs':

        fig, axs = plt.subplots()

        axs.clear()
        x = np.arange(len(data_probs_final))
        axs.bar(x, data_probs_final.values(), width=0.2, color= 'k' , align='center')
        axs.bar(x-(0.2*(0+1)), born_final_probs.values(), width=0.2, color='b', align='center')

        axs.legend((r'$\mathsf{IQP}$ Data',r'$\mathsf{QAOA}$ $\mathsf{IBM}$ with Sinkhorn.'), fontsize = 20)

        axs.set_xticks(range(len(data_probs_final)))
        axs.set_xticklabels(list(data_probs_final.keys()),rotation=70)

        plt.show()
    return

'''TWO QUBITS'''

# N_epochs = 200
# learning_rate =  0.002
# data_type = 'Quantum_Data'
# data_circuit = 'IQP'
# N_born_samples = 500
# N_data_samples = 500
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

# N_epochs = 200
# learning_rate =  0.005
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
# runs = 0

# PlotAutomaticCompilation(N_epochs, learning_rate, data_type, data_circuit,
# 							N_born_samples, N_data_samples, N_kernel_samples,
# 							batch_size, kernel_type, cost_func, qc, score,
# 							stein_eigvecs, stein_eta, sinkhorn_eps, runs, 'probs', legend = True)


def CompareCostFunctionsonQPU(N_epochs, learning_rate, data_type, data_circuit,
                        N_born_samples, N_data_samples, N_kernel_samples,
                        batch_size, kernel_type, cost_func, qc, score,
                        stein_eigvecs, stein_eta, sinkhorn_eps, runs, comparison, legend = True):
    loss, born_final_probs, data_probs_final = ReadFromFile(N_epochs, learning_rate, data_type, data_circuit,
															N_born_samples, N_data_samples, N_kernel_samples,
															batch_size, kernel_type, cost_func, qc, score,
															stein_eigvecs, stein_eta, sinkhorn_eps, runs)

    
    N_trials = len(N_epochs)
    if comparison.lower() == 'probs':
        plot_colour = ['g*-', 'y*-', 'bo-', 'co-']

        fig, axs = plt.subplots()
        data_plot_colour = 'k'


        bar_plot_colour = ['g','y', 'b', 'c']
        patterns = [ "o" , ""]

        axs.clear()
        #Plot Data
        x = np.arange(len(data_probs_final[0]))
        axs.bar(x, data_probs_final[0].values(), width=0.1, color= '%s' %data_plot_colour, align='center')
        #Plot MMD One
        axs.bar(x-(0.2*(0+1)), born_final_probs[1].values(), width=0.1, color='%s' %(bar_plot_colour[0]), align='center')
        axs.bar(x-(0.2*(0+0.5)), born_final_probs[0].values(), width=0.1, color='%s' %(bar_plot_colour[1]), align='center')

        #Plot Sinkhorn
        axs.bar(x-(0.2*(0+2)),      born_final_probs[3].values(), width=0.1, color='%s' %(bar_plot_colour[2]),   align='center')
        axs.bar(x-(0.2*(0+1.5)),    born_final_probs[2].values(), width=0.1, color='%s' %(bar_plot_colour[3]),    align='center')

        # axs.set_xlabel("Outcomes", fontsize=20)
        # axs.set_ylabel("Probability", fontsize=20)
    
        axs.legend(('Data',r'\textsf{MMD}, on %s' %qc[0], r'\textsf{MMD}, on %s' %qc[1], r'Sinkhorn, on %s' %qc[2], r'Sinkhorn, on %s' %qc[3] ), fontsize = 20)

        axs.set_xticks(range(len(data_probs_final[0])))
        axs.set_xticklabels(list(data_probs_final[0].keys()),rotation=70)
    
        plt.show()
    elif comparison.lower() == 'tv':
        plot_colour = ['g*-', 'y*-', 'bo-', 'co-']

        fig, ax = plt.subplots()

        for trial in range(N_trials):
        
            average_loss, upper_error, lower_error = AverageCostsFromFile(N_epochs[trial], learning_rate[trial], data_type[trial], data_circuit[trial],	
                                                                                N_born_samples[trial], N_data_samples[trial], N_kernel_samples[trial],
                                                                                batch_size[trial], kernel_type[trial], cost_func[trial], qc[trial], score[trial],
                                                                                stein_eigvecs[trial], stein_eta[trial], sinkhorn_eps[trial])
            cost_error = np.vstack((lower_error['TV'], upper_error['TV'])) #Stack errors into (2d, N_epochs) array for numpy errorbar function
    
            x = np.arange(0, N_epochs[trial]-1, 1)


            if cost_func[trial].lower() == 'mmd':

                ax.plot(x, average_loss['TV'], '%s' %(plot_colour[trial]), label =r'\textsf{MMD}, on %s.' % qc[trial] )

                ax.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor='%s' %plot_colour[trial][0])

                # axins.plot(x, average_loss['TV'], 'r%s-' %plot_markers[trial], \
                #         label =r'MMD for $\kappa_{%s}$, $\eta_{init}$ = %.3f.' %(kernel_type[trial][0], learning_rate[trial]) )

                
            elif cost_func[trial].lower() == 'sinkhorn':

                x_sink = np.arange(0, len(average_loss['TV']))

                ax.plot(x_sink, average_loss['TV'], '%s' %(plot_colour[trial]), label =r'Sinkhorn, on %s.' %qc[trial])

                ax.fill_between(x, average_loss['TV'] - lower_error['TV'], average_loss['TV'] + upper_error['TV'], alpha=0.2, facecolor='%s' %plot_colour[trial][0])

        plt.legend(loc='best', prop={'size': 20})
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
                        plot_colour = ['g', 'y']
                
                elif qc[trial].lower()[8] == '4':
                        plot_colour = ['g', 'y']

                x_mmd = np.arange(0, len(average_loss['MMD', 'Train']))


                plt.plot(x_mmd, average_loss['MMD', 'Train'],'%s*-' % plot_colour[trial],\
                                label =r'\textsf{MMD} on training set for %s'  %qc[trial] )
                plt.fill_between(x_mmd, average_loss['MMD', 'Train'] - lower_error['MMD', 'Train'],\
                                        average_loss['MMD', 'Train'] + upper_error['MMD', 'Train'], alpha=0.5, facecolor='%s' %plot_colour[trial])

                plt.plot(x_mmd, average_loss['MMD', 'Test'],'%s-' % plot_colour[trial],\
                                label =r'\textsf{MMD} on test set for %s' %qc[trial] )
                plt.fill_between(x_mmd, average_loss['MMD', 'Test'] - lower_error['MMD', 'Test'],\
                                        average_loss['MMD', 'Test'] + upper_error['MMD', 'Test'], alpha=0.3, facecolor='%s' %plot_colour[trial])

            #         x_mmd = np.arange(0, len(average_loss['MMD', 'Train']))

            #         plt.errorbar(x_mmd, average_loss[('MMD', 'Train')], train_error, None,\
            #                                         '%sx-' %(plot_colour[trial]), label =r'MMD on %s.' \
            #                                         %qc[trial], capsize=1, elinewidth=1, markeredgewidth=2)
            #         plt.errorbar(x_mmd, average_loss[('MMD', 'Test')], test_error, None,\
            #                                         '%s-' %(plot_colour[trial]), label =r'MMD on %s.' \
            #                                         %(qc[trial]),capsize=1, elinewidth=1, markeredgewidth=2)
            #   r'MMD Loss $\mathcal{L}_{\mathsf{MMD}}$', fontsize = 20)
            
           
            elif cost_func[trial].lower() == 'sinkhorn': 
                if qc[trial].lower()[8] == '3':
                    plot_colour = ['b', 'c']
                elif qc[trial].lower()[8] == '4':
                    plot_colour = ['b', 'c']

                    # plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Train')], train_error, None,\
                    #                                 '%sx-' %(plot_colour[trial]), label =r'Sinkhorn, on %s.' \
                    #                                 %(qc[trial]),\
                    #                                 capsize=1, elinewidth=1, markeredgewidth=2)
                    # plt.errorbar(x_sink, average_loss[('Sinkhorn', 'Test')], test_error, None,\
                    #                                 '%s-' %(plot_colour[trial]), label =r'Sinkhorn, on %s.' \
                    #                                 %(qc[trial]),\
                    #                                 capsize=1, elinewidth=1, markeredgewidth=2)
                x_sink = np.arange(0, len(average_loss['Sinkhorn', 'Train']))

                plt.plot(x_sink, average_loss['Sinkhorn', 'Train'],'%so-' % plot_colour[trial],\
                                label =r'Sinkhorn on training set for %s'  %qc[trial] )
                plt.fill_between(x_sink, average_loss['Sinkhorn', 'Train'] - lower_error['Sinkhorn', 'Train'],\
                                        average_loss['Sinkhorn', 'Train'] + upper_error['Sinkhorn', 'Train'], alpha=0.5, facecolor='%s'%plot_colour[trial])

                plt.plot(x_sink, average_loss['Sinkhorn', 'Test'],'%s-' % plot_colour[trial],\
                                label =r'Sinkhorn on test set for %s' %qc[trial] )
                plt.fill_between(x_sink, average_loss['Sinkhorn', 'Test'] - lower_error['Sinkhorn', 'Test'],\
                                        average_loss['Sinkhorn', 'Test'] + upper_error['Sinkhorn', 'Test'], alpha=0.3, facecolor='%s'%plot_colour[trial])

        plt.legend(loc='best', prop={'size': 20})

        plt.show()
    return

[N_epochs, learning_rate, data_type, data_circuit, N_born_samples, N_data_samples, N_kernel_samples, batch_size, kernel_type, \
cost_func, qc, score, stein_eigvecs, stein_eta, sinkhorn_eps, runs] = [[] for _ in range(16)] 


'''#################################'''
'''ON CHIP ASPEN-4-3Q-A'''
'''#################################'''

# N_epochs.append(100)
# learning_rate.append(0.01)
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
# learning_rate.append(0.01)
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
# learning_rate.append(0.01)
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
# sinkhorn_eps.append(0.2)
# runs.append(0)

# N_epochs.append(100)
# learning_rate.append(0.01)
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
# sinkhorn_eps.append(0.3)
# runs.append(0)


# CompareCostFunctionsonQPU(N_epochs, learning_rate, data_type, data_circuit,
#                         N_born_samples, N_data_samples, N_kernel_samples,
#                         batch_size, kernel_type, cost_func, qc, score,
#                         stein_eigvecs, stein_eta, sinkhorn_eps, runs, 'cost',  legend =True)

'''#################################'''
'''ON CHIP ASPEN-4-4Q-A'''
'''#################################'''

# N_epochs.append(100)
# learning_rate.append(0.01)
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
# learning_rate.append(0.01)
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
# learning_rate.append(0.01)
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
# sinkhorn_eps.append(0.08)
# runs.append(0)

# N_epochs.append(100)
# learning_rate.append(0.01)
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
# sinkhorn_eps.append(0.08)
# runs.append(0)

# CompareCostFunctionsonQPU(N_epochs, learning_rate, data_type, data_circuit,
#                         N_born_samples, N_data_samples, N_kernel_samples,
#                         batch_size, kernel_type, cost_func, qc, score,
#                         stein_eigvecs, stein_eta, sinkhorn_eps, runs, 'tv',  legend =True)
