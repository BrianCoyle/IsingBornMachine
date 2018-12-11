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

plot_colour = ['r', 'b']
# plot_colour.append(('r', 'b'))

## This function gathers inputs from file
#
# @param[in] file_name name of file to gather inputs from
# 
# @param[out] N_epochs number of epochs
# @param[out] N_qubits number of qubits
# @param[out] learning_rate
# @param[out] N_data_samples
# @param[out] N_born_samples
# @param[out] N_kernel_samples
# @param[out] batch_size
# @param[out] kernel_type
# @param[out] approx
# @param[out] cost_func
# @param[out] stein_approx
# @param[out] device_name
# @param[out] as_qvm_value
#
# @returns listed parameters
def get_inputs(file_name):
    
    with open(file_name, 'r') as input_file:
   
        input_values = input_file.readlines()

        N_epochs = int(input_values[0])
        
        learning_rate = float(input_values[1])

        N_data_samples = int(input_values[2])
        
        N_born_samples = int(input_values[3])

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
                
        device_name = str(input_values[10])
        device_name = device_name[0:len(device_name) - 1]

        if int(input_values[11]) == 1:
            as_qvm_value = True
        else:
            as_qvm_value = False
    
    return N_epochs, learning_rate, N_data_samples, N_born_samples, N_kernel_samples, batch_size, kernel_type, approx, cost_func, stein_approx, device_name, as_qvm_value

def SaveAnimation(framespersec, fig, N_epochs, N_qubits, learning_rate, N_born_samples, cost_func, kernel_type, approx, data_exact_dict, born_probs_list, axs, N_data_samples):

        Writer = animation.writers['ffmpeg']

        writer = Writer(fps=framespersec, metadata=dict(artist='Me'), bitrate=-1)
        
        ani = animation.FuncAnimation(fig, animate, frames=len(born_probs_list), fargs=(N_qubits, learning_rate, N_born_samples, kernel_type, approx, data_exact_dict, born_probs_list, axs, N_data_samples), interval = 10)
        
        ani.save("animations/%s_%iNv_%s_%s_%.4fLR_%iSamples_%iEpochs.mp4" \
                %(cost_func[0], N_qubits, kernel_type[0][0], approx[0][0], learning_rate, N_born_samples, N_epochs))

        plt.show()

def PlotAnimate(N_qubits, N_epochs, learning_rate, N_born_samples, cost_func, kernel_type, approx, data_exact_dict):
        
        plt.legend(prop={'size': 7}, loc='best').draggable()
        
        plt.savefig("plots/%s_%iNv_%s_%s_%iBSamps_%.3fLR_%iEpoch.pdf" \
                %(cost_func[0], N_qubits, kernel_type[0][0], approx[0][0], N_born_samples, learning_rate, N_epochs))
        
        fig, axs = plt.subplots()
        
        axs.set_xlabel("Outcomes")
        axs.set_ylabel("Probability")
        axs.legend(('Born Probs','Data Probs'))
        axs.set_xticks(range(len(data_exact_dict)))
        axs.set_xticklabels(list(data_exact_dict.keys()),rotation=70)
        axs.set_title("%i Qubits, %s Kernel, %s Learning Rate = %.4f, %i Born Samples" \
                %(N_qubits, kernel_type[0][0], approx[0][0], learning_rate, N_born_samples))
        
        plt.tight_layout()

        return fig, axs

def animate(i, N_qubits, learning_rate, N_born_samples, kernel_type, approx, data_exact_dict, born_probs_list, axs, N_data_samples):
    
    axs.clear()
    x = np.arange(len(data_exact_dict))
    axs.bar(x, born_probs_list[i].values(), width=0.2, color= plot_colour[0], align='center')
    axs.bar(x-0.2, data_exact_dict.values(), width=0.2, color='b', align='center')
    axs.set_title("%i Qbs, %s Kernel, %s Learning Rate = %.4f, %i Data Samps, %i Born Samps" \
                %(N_qubits, kernel_type[0][0], approx[0][0], learning_rate, N_data_samples, N_born_samples))
    axs.set_xlabel("Outcomes")
    axs.set_ylabel("Probability")
    axs.legend(('Born Probs','Data Probs'))
    axs.set_xticks(range(len(data_exact_dict)))
    axs.set_xticklabels(list(data_exact_dict.keys()),rotation=70)

## This is the main function
def main():

    if len(sys.argv) != 2:
        sys.exit("[ERROR] : There should be exactly one input. Namely, a txt file containing the input values")
    else:
        N_epochs, learning_rate, N_data_samples, N_born_samples, N_kernel_samples, batch_size, kernel_type, approx, cost_func, stein_approx, device_name, as_qvm_value = get_inputs(sys.argv[1])

        device_params = [device_name, as_qvm_value]
        qc = get_qc(device_name, as_qvm = as_qvm_value)
        qubits = qc.qubits()
        N_qubits = len(qubits)
        #Parameters, J, b for epoch 0 at random, gamma = constant = pi/4

        initial_params = NetworkParams(device_params)


        '''Number of samples:'''
        N_samples =     [N_data_samples,\
                        N_born_samples,\
                        batch_size,\
                        N_kernel_samples]
        
        data_samples, data_exact_dict = DataImport(approx, N_qubits, N_data_samples, stein_approx)

    #Randomise data
    np.random.shuffle(data_samples)
    #Split data into training/test sets
    train_test = TrainTestPartition(data_samples)

    plt.figure(1)
    
    loss, circuit_params, born_probs_list, empirical_probs_list  = CostPlot(device_params, N_epochs, initial_params, learning_rate,\
                                                                                    approx, kernel_type, train_test, data_exact_dict,\
                                                                                    N_samples, plot_colour,\
                                                                                    cost_func, stein_approx, 'Precompute')
    
    fig, axs = PlotAnimate(N_qubits, N_epochs, learning_rate, N_born_samples, cost_func, kernel_type, approx, data_exact_dict)

    SaveAnimation(2000, fig, N_epochs, N_qubits, learning_rate, N_born_samples, cost_func, kernel_type, approx, data_exact_dict, born_probs_list, axs, N_data_samples)
    
    console_output = sys.stdout
    sys.stdout = open("Output_%sCost_%sDevice_%skernel_%iN_k_samples_%i_N_Born_Samples%iN_Data_samples_%iBatch_size_%iEpochs_%.3flr" \
                %(cost_func,\
                device_params[0],\
                kernel_type,\
                N_kernel_samples,\
                N_born_samples,\
                N_data_samples,\
                batch_size,\
                N_epochs,\
                learning_rate), 'w')
    PrintFinalParamsToFile(cost_func, N_epochs, loss, circuit_params, born_probs_list, empirical_probs_list, device_params, kernel_type, N_samples, learning_rate)
    sys.stdout.close()
    sys.stdout= console_output

if __name__ == "__main__":

    main() 