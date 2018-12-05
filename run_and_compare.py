## @package run_and_compare
# This is the main module for this project.
#
# More details.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, style
from pyquil.api import get_qc
from param_init import NetworkParams
from file_operations_out import PrintFinalParamsToFile
from file_operations_in import DataImport
from train_plot import CostPlot
from random import shuffle
from auxiliary_functions import TrainTestPartition
import sys

plot_colour = []
plot_colour.append(('r', 'b'))
plot_colour.append(('m', 'c'))

## This function gathers inputs from file
#
# @param[in] file_name name of file to gather inputs from
# 
# @param[out] N_epochs number of epochs
# @param[out] N_qubits number of qubits
# @param[out] learning_rate_one first learning rate
# @param[out] learning_rate_two second learning rate
#
# @returns listed parameters
def get_inputs(file_name):
    
    with open(file_name, 'r') as input_file:
   
        input_values = input_file.readlines()

        N_epochs = int(input_values[0])
        learning_rate_one = float(input_values[1])
        learning_rate_two = float(input_values[2])
        N_data_samples = int(input_values[3])
        N_kernel_samples = int(input_values[4])
        batch_size = int(input_values[5])
        kernel_type = str(input_values[6])
        kernel_type = kernel_type[0:len(kernel_type) - 1]
        approx = str(input_values[7])
        approx = approx[0:len(approx) - 1]
        cost_func = str(input_values[8])
        cost_func = cost_func[0:len(cost_func) - 1]
        stein_approx = str(input_values[9])
        stein_approx = stein_approx[0:len(stein_approx) - 1]
        weight_sign = int(input_values[10])
        device_name = str(input_values[11])
        device_name = device_name[0:len(device_name) - 1]

        if int(input_values[12]) == 1:
            as_qvm_value = True
        else:
            as_qvm_value = False
    
    return N_epochs, learning_rate_one, learning_rate_two, N_data_samples, N_kernel_samples, batch_size, kernel_type, approx, cost_func, stein_approx, weight_sign, device_name, as_qvm_value

def SaveAnimation(framespersec, fig, N_epochs, N_qubits, learning_rate, N_born_samples, cost_func, kernel_type, approx, data_exact_dict1, born_probs_list1, axs, N_data_samples, born_probs_list2):

        Writer = animation.writers['ffmpeg']

        writer = Writer(fps=framespersec, metadata=dict(artist='Me'), bitrate=-1)
        
        ani = animation.FuncAnimation(fig, animate, frames=len(born_probs_list1), fargs=(N_qubits, learning_rate, N_born_samples, kernel_type, approx, data_exact_dict1, born_probs_list1, axs, N_data_samples, born_probs_list2), interval = 10)
        
        ani.save("%s_%iNv_%s_%s_%.4fLR_%iSamples_%iEpochs.mp4" \
                %(cost_func[0], N_qubits, kernel_type[0][0], approx[0][0], learning_rate[0], N_born_samples[0], N_epochs))

        plt.show()

def PlotAnimate(N_qubits, N_epochs, learning_rate, N_born_samples, cost_func, kernel_type, approx, data_exact_dict1):
        
        plt.legend(prop={'size': 7}, loc='best').draggable()
        
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

        return fig, axs

def animate(i, N_qubits, learning_rate, N_born_samples, kernel_type, approx, data_exact_dict1, born_probs_list1, axs, N_data_samples, born_probs_list2):
    
    axs.clear()
    x = np.arange(len(data_exact_dict1))
    axs.bar(x, born_probs_list1[i].values(), width=0.2, color= plot_colour[0][0], align='center')
    axs.bar(x-0.2, data_exact_dict1.values(), width=0.2, color='b', align='center')
    axs.set_title("%i Qbs, %s Kernel, %s Learning Rate = %.4f, %i Data Samps, %i Born Samps" \
                %(N_qubits, kernel_type[0][0], approx[0][0], learning_rate[0], N_born_samples[1], N_born_samples[0]))
    axs.set_xlabel("Outcomes")
    axs.set_ylabel("Probability")
    axs.legend(('Born Probs','Data Probs'))
    axs.set_xticks(range(len(data_exact_dict1)))
    axs.set_xticklabels(list(data_exact_dict1.keys()),rotation=70)

## This is the main function
def main():

    if len(sys.argv) != 2:
        sys.exit("[ERROR] : There should be exactly one input. Namely, a txt file containing the input values")
    else:
        N_epochs, learning_rate_one, learning_rate_two, N_data_samples, N_kernel_samples, batch_size, kernel_type, approx, cost_func, stein_approx, weight_sign, device_name, as_qvm_value = get_inputs(sys.argv[1])

        device_params = [device_name, as_qvm_value]
        qc = get_qc(device_name, as_qvm = as_qvm_value)
        qubits = qc.qubits()
        N_qubits = len(qubits)
        #Parameters, J, b for epoch 0 at random, gamma = constant = pi/4

        initial_params = NetworkParams(device_params)

        #Set learning rate for parameter updates
        learning_rate = [learning_rate_one, learning_rate_two] 

        '''If kernel is to be computed exactly set N_kernel_samples = 'infinite' '''
        N_born_samples =        [50, 100]
        N_bornplus_samples =    N_born_samples
        N_bornminus_samples =   N_born_samples

        N_samples = {}

        '''Trial 1 Number of samples:'''
        N_samples['Trial_1'] = [N_data_samples,\
                                N_born_samples[0],\
                                N_bornplus_samples[0],\
                                N_bornminus_samples[0],\
                                N_kernel_samples]
        
        data_samples1, data_exact_dict1 = DataImport(approx, N_qubits, N_data_samples, stein_approx)

    #Randomise data
    np.random.shuffle(data_samples1)

    #Split data into training/test sets
    train_test_1 = TrainTestPartition(data_samples1)

    plt.figure(1)
    
    loss1, circuit_params1, born_probs_list1, empirical_probs_list1  = CostPlot(device_params, N_epochs, initial_params, learning_rate[0],\
                                                                                    approx, kernel_type, train_test_1, data_exact_dict1,\
                                                                                    N_samples['Trial_1'], plot_colour[0], weight_sign,\
                                                                                    cost_func, stein_approx, 'Onfly', batch_size)
    
    fig, axs = PlotAnimate(N_qubits, N_epochs, learning_rate, N_born_samples, cost_func, kernel_type, approx, data_exact_dict1)

    SaveAnimation(2000, fig, N_epochs, N_qubits, learning_rate, N_born_samples, cost_func, kernel_type, approx, data_exact_dict1, born_probs_list1, axs, N_data_samples, born_probs_list1)

if __name__ == "__main__":

    main() 
                
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
