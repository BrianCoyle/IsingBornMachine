## @package run_and_compare
# This is the main module for this project.
#
# More details.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, style
from pyquil.api import get_qc
from param_init import NetworkParams
from file_operations_out import PrintFinalParamsToFile, PrintDataToFiles, MakeDirectory, PrintFinalParamsToFile
from file_operations_in import DataImport, DataDictFromFile
from train_plot import CostPlot
from random import shuffle
from auxiliary_functions import TrainTestPartition, FindNumQubits, SampleListToArray
import sys
import os

## This function gathers inputs from file
#
# @param[in] file_name name of file to gather inputs from
# 
# @param[out] N_epochs number of epochs
# @param[out] data_type
# @param[out] N_data_samples
# @param[out] N_born_samples
# @param[out] N_kernel_samples
# @param[out] batch_size
# @param[out] kernel_type
# @param[out] cost_func
# @param[out] device_name
# @param[out] as_qvm_value
#
# @returns listed parameters
def get_inputs(file_name):
    
    with open(file_name, 'r') as input_file:
   
        input_values = input_file.readlines()

        N_epochs = int(input_values[0])
        
        data_type = str(input_values[1])
        data_type = data_type[0:len(data_type) - 1]

        N_data_samples = int(input_values[2])
        
        N_born_samples = int(input_values[3])

        N_kernel_samples = int(input_values[4])
        
        batch_size = int(input_values[5])
        
        kernel_type = str(input_values[6])
        kernel_type = kernel_type[0:len(kernel_type) - 1]
        
        cost_func = str(input_values[7])
        cost_func = cost_func[0:len(cost_func) - 1]
        
        device_name = str(input_values[8])
        device_name = device_name[0:len(device_name) - 1]

        if int(input_values[9]) == 1:
            as_qvm_value = True
        else:
            as_qvm_value = False

    
    return N_epochs, data_type, N_data_samples, N_born_samples, N_kernel_samples, batch_size, kernel_type, cost_func, device_name, as_qvm_value

def SaveAnimation(framespersec, fig, N_epochs, N_qubits, N_born_samples, cost_func, kernel_type, data_exact_dict, born_probs_list, axs, N_data_samples):

        Writer = animation.writers['ffmpeg']

        writer = Writer(fps=framespersec, metadata=dict(artist='Me'), bitrate=-1)
        
        ani = animation.FuncAnimation(fig, animate, frames=len(born_probs_list), fargs=(N_qubits, N_born_samples, kernel_type, data_exact_dict, born_probs_list, axs, N_data_samples), interval = 10)
        
        ani.save("animations/%s_%iQbs_%s_Kernel_%iSamples_%iEpochs.mp4" \
                %(cost_func[0], N_qubits, kernel_type[0][0], N_born_samples, N_epochs))

        plt.show()

def PlotAnimate(N_qubits, N_epochs, N_born_samples, cost_func, kernel_type, data_exact_dict):
        
        plt.legend(prop={'size': 7}, loc='best').draggable()
        
        plt.savefig("plots/%s_%iQbs_%s_%iBSamps_%iEpoch.pdf" \
                %(cost_func[0], N_qubits, kernel_type[0][0], N_born_samples, N_epochs))
        
        fig, axs = plt.subplots()
        
        axs.set_xlabel("Outcomes")
        axs.set_ylabel("Probability")
        axs.legend(('Born Probs','Data Probs'))
        axs.set_xticks(range(len(data_exact_dict)))
        axs.set_xticklabels(list(data_exact_dict.keys()),rotation=70)
        axs.set_title("%i Qubits, %s Kernel, %i Born Samples" \
                %(N_qubits, kernel_type[0][0], N_born_samples))
        
        plt.tight_layout()

        return fig, axs

def animate(i, N_qubits, N_born_samples, kernel_type,  data_exact_dict, born_probs_list, axs, N_data_samples):
        plot_colour = ['r', 'b']
        axs.clear()
        x = np.arange(len(data_exact_dict))
        axs.bar(x, born_probs_list[i].values(), width=0.2, color= plot_colour[0], align='center')
        axs.bar(x-0.2, data_exact_dict.values(), width=0.2, color='b', align='center')
        axs.set_title("%i Qbs, %s Kernel, %i Data Samps, %i Born Samps" \
                %(N_qubits, kernel_type[0][0], N_data_samples, N_born_samples))
        axs.set_xlabel("Outcomes")
        axs.set_ylabel("Probability")
        axs.legend(('Born Probs','Data Probs'))
        axs.set_xticks(range(len(data_exact_dict)))
        axs.set_xticklabels(list(data_exact_dict.keys()),rotation=70)

def bytes_to_int(bytes_list):

    total = 0

    for byte in bytes_list:

        total *= 256
        total += byte

    return total

def num_bytes_needed(num_bits):

    num_bytes = num_bits // 8

    if num_bits % 8 != 0:
        num_bytes += 1

    return num_bytes

def read_ints_from_file(N_qubits, N_data_samples, f):

    int_list = [0] * N_data_samples

    bytes_list = list(f.read())

    for sample in range(N_data_samples):

        int_list[sample] = bytes_to_int(bytes_list[sample * num_bytes_needed(N_qubits):(sample + 1) * num_bytes_needed(N_qubits)])

    return int_list


## This is the main function
def main():

    if len(sys.argv) != 2:
        sys.exit("[ERROR] : There should be exactly one input. Namely, a txt file containing the input values. Please see the README.md file for more details.")
    else:
        N_epochs, data_type, N_data_samples, N_born_samples, N_kernel_samples, batch_size, kernel_type, cost_func, device_name, as_qvm_value = get_inputs(sys.argv[1])
       
        device_params = [device_name, as_qvm_value]
        N_qubits = FindNumQubits(device_params)
        circuit_type = 'QAOA'
 
        if data_type == 'Quantum_Data':

            try:
                data_samples_orig = list(np.loadtxt('binary_data/Quantum_Data_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_data_samples, circuit_type), dtype = str))
            except:
                PrintDataToFiles(data_type, N_data_samples, device_params, circuit_type, N_qubits)

                data_samples_orig = list(np.loadtxt('binary_data/Quantum_Data_%iQBs_%iSamples_%sCircuit' % (N_qubits, N_data_samples, circuit_type), dtype = str))

        elif data_type == 'Classical_Data':
            
            try:
    
                with open('binary_data/Classical_Data_%iQBs_%iSamples' % (N_qubits, N_data_samples), 'rb') as f:

                    data_samples_orig = read_ints_from_file(N_qubits, N_data_samples, f)

            except:

                PrintDataToFiles(data_type, N_data_samples, device_params, circuit_type, N_qubits)

                with open('binary_data/Classical_Data_%iQBs_%iSamples' % (N_qubits, N_data_samples), 'rb') as f:

                    data_samples_orig = read_ints_from_file(N_qubits, N_data_samples, f)

        else:
            sys.exit("[ERROR] : data_type should be either 'Quantum_Data' or 'Classical_Data'")

        data_samples = np.zeros((N_data_samples, N_qubits), dtype = int)

        for sample in range(0, N_data_samples):
            
            temp = data_samples_orig[sample]

            for outcome in range(0, N_qubits):

                data_samples[sample, outcome] = temp % 2
                temp >>= 1

        np.random.shuffle(data_samples)

        #Split data into training/test sets
        data_train_test = TrainTestPartition(data_samples)

        plt.figure(1)
 
        random_seed = 0

        #Parameters, J, b for epoch 0 at random, gamma = constant = pi/4
        #Set random seed to 0 to initialise the actual Born machine to be trained
        initial_params = NetworkParams(device_params, random_seed)

        '''Number of samples:'''
        N_samples =     [N_data_samples,\
                        N_born_samples,\
                        batch_size,\
                        N_kernel_samples]

        data_exact_dict = DataDictFromFile(data_type, N_qubits, 'infinite', N_data_samples, circuit_type)
  
        loss, circuit_params, born_probs_list, empirical_probs_list  = CostPlot(device_params, N_epochs, initial_params, \
                                                                                    kernel_type,\
                                                                                    data_train_test, data_exact_dict, \
                                                                                    N_samples,\
                                                                                    cost_func, 'Onfly')
   
        fig, axs = PlotAnimate(N_qubits, N_epochs, N_born_samples, cost_func, kernel_type, data_exact_dict)
        SaveAnimation(2000, fig, N_epochs, N_qubits,  N_born_samples, cost_func, kernel_type, data_exact_dict, born_probs_list, axs, N_data_samples)
        
        path_to_output = './outputs/Output_%sCost_%sDevice_%skernel_%ikernel_samples_%iBorn_Samples%iData_samples_%iBatch_size_%iEpochs/'  \
                                                         %(cost_func,\
                                                         device_params[0],\
                                                         kernel_type,\
                                                         N_kernel_samples,\
                                                         N_born_samples,\
                                                         N_data_samples,\
                                                         batch_size,\
                                                         N_epochs)


        PrintFinalParamsToFile(cost_func, N_epochs, loss, circuit_params, born_probs_list, empirical_probs_list, device_params, kernel_type, N_samples)

if __name__ == "__main__":

    main() 
