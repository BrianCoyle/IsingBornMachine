import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import style

from param_init import NetworkParams

from file_operations_out import PrintFinalParamsToFile

from file_operations_in import DataImport

from train_plot import CostPlot
from random import shuffle

from auxiliary_functions import TrainTestPartition
import sys
#N_epoch is the total number of training epochs
N_epochs = 100
#N is the total number of qubits.
N_qubits = 2
N_trials = 1

# #Initialise a 3 dim array for the graph weights, 2 dim array for biases and gamma parameters
# J = np.zeros((N1, N1))
# b = np.zeros((N1))
# gamma_x = np.zeros((N1))
# gamma_y = np.zeros((N1))

#Parameters, J, b for epoch 0 at random, gamma = constant = pi/4
#Read in from file for reproducibility

initial_params = {}
initial_params['J'], initial_params['b'], \
initial_params['gamma_x'], initial_params['gamma_y'] = NetworkParams(N_qubits)
# print(J_i, b_i, g_x_i, g_y_i)
#Set learning rate for parameter updates
learning_rate = []
learning_rate.append(0.05)
learning_rate.append(0.05)
# learning_rate[2] = 0.001

weight_sign = []
weight_sign.append(-1)
weight_sign.append(-1)
weight_sign.append(-1)

batch_size = []

'''If kernel is to be computed exactly set N_kernel_samples = 'infinite' '''
N_data_samples =        []
N_born_samples =        []
N_bornplus_samples=     []
N_bornminus_samples=    []
N_kernel_samples=       []
N_samples=              {}

'''Trial 1 Number of samples:'''
N_data_samples.append(100)
N_born_samples.append(100)
N_bornplus_samples.append(N_born_samples[0])
N_bornminus_samples.append(N_born_samples[0])
N_kernel_samples.append(2000)


N_samples['Trial_1'] = [N_data_samples[0],\
                        N_born_samples[0],\
                        N_bornplus_samples[0],\
                        N_bornminus_samples[0],\
                        N_kernel_samples[0]]
# batch_size.append(N_data_samples[0])
batch_size.append(10)

'''Trial 2 Number of samples:'''
N_data_samples.append(100)
N_born_samples.append(100)
N_bornplus_samples.append(N_born_samples[1])
N_bornminus_samples.append(N_born_samples[1])
N_kernel_samples.append(2000)
if (N_trials == 2):
        N_samples['Trial_2'] = [N_data_samples[1], \
                                N_born_samples[1],\
                                N_bornplus_samples[1],\
                                N_bornminus_samples[1],\
                                N_kernel_samples[1]]
        batch_size.append(10)

'''Trial 3 Number of samples:'''
N_born_samples.append(1)
N_bornplus_samples.append(N_born_samples[2])
N_bornminus_samples.append(N_born_samples[2])
N_kernel_samples.append(2000)

if (N_trials == 3):
        N_samples['Trial_3'] = [N_data_samples[2], \
                                N_born_samples[2],\
                                N_bornplus_samples[2],\
                                N_bornminus_samples[2],\
                                N_kernel_samples[2]]
        batch_size.append(10)


kernel_type = []
kernel_type.append('Gaussian')
kernel_type.append('Gaussian')
kernel_type.append('Gaussian')

approx = []
approx.append('Sampler')
approx.append('Sampler')
approx.append('Exact')

cost_func = []
cost_func.append('MMD')
cost_func.append('Stein')

stein_approx = []
stein_approx.append('Exact_Score')
stein_approx.append('Exact_Score')
# stein_approx.append('Exact_Score')

data_samples1, data_exact_dict1 = DataImport(approx[0], N_qubits, N_data_samples[0], stein_approx[0])
data_samples2, data_exact_dict2 = DataImport(approx[1], N_qubits, N_data_samples[1], stein_approx[1])
# print(data_samples1)
# print(data_samples2)

#Randomise data
np.random.shuffle(data_samples1)
np.random.shuffle(data_samples2)
#Split data into training/test sets
train_test_1 = TrainTestPartition(data_samples1)
train_test_2 = TrainTestPartition(data_samples1)


# data_samples, data_exact_dict2 = DataImport(approx[1], N_v2, N_data_samples)
# data_samples, data_exact_dict3 = DataImport(approx[2], N_v3, N_data_samples)
plt.figure(1)
plot_colour = []
plot_colour.append(('r', 'b'))
plot_colour.append(('m', 'c'))

loss1, circuit_params1, born_probs_list1, empirical_probs_list1  = CostPlot(N_qubits, N_epochs, initial_params, learning_rate[0],\
                                                                                approx[0], kernel_type[0], train_test_1, data_exact_dict1,\
                                                                                N_samples['Trial_1'], plot_colour[0], weight_sign[0],\
                                                                                cost_func[0], stein_approx[0], 'Onfly', batch_size[0])

# loss2, circuit_params2, born_probs_list2, empirical_probs_list2 = CostPlot(N_qubits, N_epochs, initial_params,learning_rate[1],\
#                                                                                 approx[1], kernel_type[1], train_test_2, data_exact_dict2, \
#                                                                                 N_samples['Trial_2'], plot_colour[1], weight_sign[1],\
#                                                                                 cost_func[1], stein_approx[1],'Precompute',batch_size[1])

# loss3, circuit_params3, born_probs_list3, empirical_probs_list3 = PlotMMD(N_qubits, N_epochs, initial_params,learning_rate[2],\
#                                                                                 approx[2], kernel_type[2], train_test_3, data_exact_dict[2], \
#                                                                                 N_samples['Trial_3'], 'b', weight_sign[2], cost_func[2], stein_approx[2])


def PlotAnimate(N_trials):
        plt.legend(prop={'size': 7}, loc='best').draggable()

        if (N_trials == 1):
        
                plt.savefig("%s_%iNv_%s_%s_%iBSamps_%.3fLR_%iEpoch.pdf" \
                %(cost_func[0], N_qubits, kernel_type[0][0], approx[0][0], N_born_samples[0], learning_rate[0], N_epochs))
                
                fig, axs = plt.subplots()
                
                axs.set_xlabel("Outcomes")
                axs.set_ylabel("Probability")
                axs.legend(('Born Probs','Data Probs'))
                axs.set_xticks(range(len(data_exact_dict1)))
                axs.set_xticklabels(list(data_exact_dict1.keys()),rotation=70)
                axs.set_title("%i Qubits, %s Kernel, %s Learning Rate = %.4f, %i Born Samples" \
                        %(N_qubits, kernel_type[0][0], approx[0][0], learning_rate[0], N_born_samples[0]))

                plt.tight_layout()

        elif (N_trials == 2):

                plt.savefig("%s_%iNv_%s_%s_%iBSamps_%.3fLR_%s_%s_%s_%iBSamps_%.3fLR_%iEpoch.pdf" \
                %(cost_func[0],N_qubits, kernel_type[0][0], approx[0][0], N_born_samples[0], learning_rate[0], cost_func[1],\
                        kernel_type[1][0], approx[1][0], N_born_samples[1], learning_rate[1], N_epochs))
                
                fig, axs = plt.subplots(2)


                for i in range(0, N_trials):
                        axs[i].set_xlabel("Outcomes")
                        axs[i].set_ylabel("Probability")
                        axs[i].legend(('Born Probs','Data Probs'))
                        axs[i].set_xticks(range(len(data_exact_dict2)))
                        axs[i].set_xticklabels(list(data_exact_dict2.keys()),rotation=70)
                        if (i == 0):
                                axs[i].set_title("%i Qubits, %s Kernel, %s Learning Rate = %.4f, %i Born Samps" \
                                        %(N_qubits, kernel_type[0][0], approx[0][0], learning_rate[0], N_born_samples[0]))
                        elif (i == 1):
                                 axs[i].set_title("%i Qubits, %s Kernel, %s Learning Rate = %.4f, %i Born Samps" \
                                        %(N_qubits, kernel_type[0][0], approx[0][0], learning_rate[0], N_born_samples[1]))
                plt.tight_layout()

        elif (N_trials == 3):
                plt.savefig("%s_%iNv_%s_%s_%iBSamps_%.3fLR_%s_%s_%s_%iBSamps_%.3fLR_%s_%s_%s_%iBSamps_%.3fLR_%iEpoch.pdf" \
                %(cost_func[0],N_qubits, kernel_type[0][0], approx[0][0], N_born_samples[0], learning_rate[0], cost_func[1],\
                        kernel_type[1][0], approx[1][0], N_born_samples[1], learning_rate[1], cost_func[2],\
                        kernel_type[2][0], approx[2][0], N_born_samples[2], learning_rate[2], N_epochs))

                fig, axs = plt.subplots(N_trials)
        
                for trial in range(0, N_trials):
                        axs[trial].set_xlabel("Outcomes")
                        axs[trial].set_ylabel("Probability")
                        axs[trial].legend(('Born Probs','Data Probs'))
                        axs[trial].set_xticks(range(len(data_exact_dict1)))
                        axs[trial].set_xticklabels(list(data_exact_dict1.keys()),rotation=70)
                        if (trial == 0):
                                axs[trial].set_title("%i Qubits, %s Kernel, %s $\eta$ = %.4f, %i Born Samps" \
                                        %(N_qubits, kernel_type[trial][0], approx[trial][0], learning_rate[trial], N_born_samples[0]))
                        elif (trial == 1):
                                 axs[trial].set_title("%i Qs, %s Kernel, %s $\eta$ = %.4f, %i Born Samps" \
                                        %(N_qubits, kernel_type[trial][0], approx[trial][0], learning_rate[trial], N_born_samples[1]))
                        elif (trial == 2):
                                 axs[trial].set_title("%i Qubits, %s Kernel, %s $\eta$ = %.4f, %i Born Samps" \
                                         %(N_qubits, kernel_type[trial][0], approx[trial][0], learning_rate[trial], N_born_samples[2]))
        
                plt.tight_layout()

        return fig, axs
fig, axs = PlotAnimate(N_trials)

def animate1(i):
      
        if N_trials is not 1:
                raise IOError('You have used the wrong animate function, please use either animate2, or animate3')
        else:
                for trial in range(0, N_trials):
                        axs.clear()
                        x = np.arange(len(data_exact_dict1))
                        axs.bar(x, born_probs_list1[i].values(), width=0.2, color= plot_colour[trial][0], align='center')
                        axs.bar(x-0.2, data_exact_dict1.values(), width=0.2, color='b', align='center')
                        axs.set_title("%i Qbs, %s Kernel, %s Learning Rate = %.4f, %i Data Samps, %i Born Samps" \
                                        %(N_qubits, kernel_type[trial][0], approx[trial][0], learning_rate[trial], N_born_samples[1], N_born_samples[0]))
        
                        axs.set_xlabel("Outcomes")
                        axs.set_ylabel("Probability")
                        axs.legend(('Born Probs','Data Probs'))
                        axs.set_xticks(range(len(data_exact_dict1)))
                        axs.set_xticklabels(list(data_exact_dict1.keys()),rotation=70)
                   
                
def animate2(i):

        if N_trials is not 2:
                raise IOError('You have used the wrong animate function, please use either animate1, or animate3')
        else:
                for trial in range(0, N_trials):
                        axs[trial].clear()
                        x = np.arange(len(data_exact_dict1))
                        axs[trial].bar(x-0.2, data_exact_dict1.values(), width=0.2, color=plot_colour[trial][0], align='center')
                        if (trial == 0):   
                                axs[trial].bar(x, born_probs_list1[i].values(), width=0.2, color='c', align='center')
                                axs[trial].title.set_text("%i Qbs, %s Kernel, %s $\eta$  = %.4f, %i Data Samps, %i Born Samps"  \
                                                %(N_qubits, kernel_type[trial], approx[trial], learning_rate[trial], N_data_samples[0], N_born_samples[0]))
                        elif (trial ==1):   
                                axs[trial].bar(x, born_probs_list2[i].values(), width=0.2, color='c', align='center')   
                                axs[trial].title.set_text("%i Qbs, %s Kernel, %s $\eta$  = %.4f, %i Data Samps, %i Born Samps" \
                                                %(N_qubits, kernel_type[trial], approx[trial], learning_rate[trial], N_data_samples[1], N_born_samples[1]))
        
                        axs[trial].set_xlabel("Outcomes")
                        axs[trial].set_ylabel("Probability")
                        axs[trial].legend(('Born Probs','Data Probs'))
                        axs[trial].set_xticks(range(len(data_exact_dict1)))
                        axs[trial].set_xticklabels(list(data_exact_dict1.keys()),rotation=70)
                        axs[trial].set_title("%i Qbs, %s Kernel, %s $\eta$ = %.4f, %i Born Samples" \
                                %(N_qubits, kernel_type[trial][0], approx[trial][0], learning_rate[trial], N_born_samples[0]))

            


def animate3(i):
      
        if N_trials is not 3:
                raise IOError('You have used the wrong animate function, please use either animate1, or animate2')
        else:
                bar_color = ['r', 'g', 'b']
                for trial in range(0, N_trials):
                        axs[trial].clear()
                        x = np.arange(len(data_exact_dict1))
                        axs[trial].bar(x, born_probs_list1[i].values(), width=0.2, color='c', align='center')
                        axs[trial].bar(x-0.2, data_exact_dict1.values(), width=0.2, color=bar_color[trial][0], align='center')
                        if (trial ==0):      
                                axs[trial].bar(x, born_probs_list1[i].values(), width=0.2, color='c', align='center')
                                axs[trial].title.set_text("%i Qbs, %s Kernel, %s $\eta$ = %.4f, %i Data Samps, %i Born Samps" \
                                                %(N_qubits, kernel_type[trial], approx[trial], learning_rate[trial], N_data_samples[0], N_born_samples[0]))
                        elif (trial==1):      
                                axs[trial].bar(x, born_probs_list2[i].values(), width=0.2, color='c', align='center')
                                axs[trial].title.set_text("%i Qbs, %s Kernel, %s $\eta$ = %.4f, %i Data Samps, %i Born Samps" \
                                                %(N_qubits, kernel_type[trial], approx[trial], learning_rate[trial], N_data_samples[1], N_born_samples[1]))
                        elif (trial ==2):      
                                axs[trial].bar(x, born_probs_list3[i].values(), width=0.2, color='c', align='center')
                                axs[trial].title.set_text("%i Qbs, %s Kernel, %s $\eta$ = %.4f, %i Data Samps, %i Born Samps" \
                                                %(N_qubits, kernel_type[trial], approx[trial], learning_rate[trial], N_data_samples[2], N_born_samples[2]))

  
                        axs[trial].set_xlabel("Outcomes")
                        axs[trial].set_ylabel("Probability")
                        axs[trial].legend(('Born Probs','Data Probs'))
                        axs[trial].set_xticks(range(len(data_exact_dict1)))
                        axs[trial].set_xticklabels(list(data_exact_dict1.keys()),rotation=70)
               
  
def SaveAnimation(N_trials, framespersec):

        if (N_trials == 1): 
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=framespersec, metadata=dict(artist='Me'), bitrate=-1)

                ani = animation.FuncAnimation(fig, animate1, frames=len(born_probs_list1),  interval = 10)
                ani.save("%s_%iNv_%s_%s_%.4fLR_%iSamples_%iEpochs.mp4" \
                        %(cost_func[0], N_qubits, kernel_type[0][0], approx[0][0], learning_rate[0], N_born_samples[0], N_epochs))

                plt.show()
        if (N_trials == 2): 
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=framespersec, metadata=dict(artist='Me'), bitrate=-1)

                ani = animation.FuncAnimation(fig, animate2, frames=len(born_probs_list1),  interval = 10)
                ani.save("%s_%iNv_%s_%s_%.4fLR_%iSamples_%s_%s_%s_%.4fLR_%iSamples_%iEpochs.mp4" \
                        %(cost_func[0], N_qubits, kernel_type[0][0], approx[0][0], learning_rate[0], N_born_samples[0], cost_func[1], kernel_type[1][0], approx[1][0],\
                        learning_rate[1], N_born_samples[1], N_epochs))
                plt.show()
        if (N_trials == 3): 
                Writer = animation.writers['ffmpeg']
                writer = Writer(fps=framespersec, metadata=dict(artist='Me'), bitrate=-1)

                ani = animation.FuncAnimation(fig, animate3, frames=len(born_probs_list1),  interval = 100)
                ani.save("%s_%iNv_%s_%s_%.4fLR_%iSamples_%s_%s_%s_%.4fLR_%iSamples_%s_%s_%s_%.4fLR_%iSamples_%iEpochs.mp4" \
                        %(cost_func[0], N_qubits, kernel_type[0][0], approx[0][0], learning_rate[0], N_born_samples[0], cost_func[1], \
                        kernel_type[1][0], approx[1][0], learning_rate[1], N_born_samples[1],cost_func[2],\
                        kernel_type[2][0], approx[2][0], learning_rate[2], N_born_samples[2], \
                        N_epochs))
                plt.show()

SaveAnimation(N_trials, 2000)

          



# console_output = sys.stdout
# sys.stdout = open("Output_MMD_%iNv_%s_%iBS_%iNv_%s_%iBS_%iNv_%s_%iBS_%iEpochs_%i_DS" \
#     %(N_qubits, kernel_type[0][0], N_born_samples[0], N_v2, kernel_type[1][0], N_born_samples[1], N_v3, kernel_type[2][0], N_born_samples[2],  N_epochs, N_data_samples), 'w')
# sys.stdout = open("Output_MMD_%iNv_%s_%s_%.3fLR_%s_%s_%.3fLR_%iEpochs" \
#     %(N_qubits, kernel_type[0][0], approx1 ,learning_rate[0], kernel_type[1][0], approx[1][0], learning_rate[1], N_epochs), 'w')
#PrintParamsToFile(J1, b1, L1, N_qubits, kernel_type[0], N_born_samples[0], N_epochs, N_born_samples[1], learning_rate[0])
#PrintParamsToFile(J2, b2, L2, N_v2, kernel_type[1], N_born_samples[1], N_epochs, N_data_samples[1], learning_rate[1])
#PrintParamsToFile(J3, b3, L3, N_v3, kernel_type[2], N_born_samples[2], N_epochs, N_data_samples[2], learning_rate[2])

# sys.stdout.close()
# sys.stdout= console_output
