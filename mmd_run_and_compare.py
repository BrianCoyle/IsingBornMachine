import numpy as np
import matplotlib.pyplot as plt
from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler
from classical_kernel import GaussianKernel
from mmd import MMDTrain, MMDKernelforGradExact, MMDKernelforGradSampler,  MMDGradSampler,MMDGradExact,  KernelComputation, EncodingFunc, MMDCost, MMDCostExact
from file_operations_out import DataDictFromFile
from mmd_train_plot import PlotMMD, DataImport, PrintParamsToFile
from param_init import StateInit, NetworkParams

import sys
#N_epoch is the total number of training epochs
N_epochs = 200
#N is the total number of qubits.
N1 = 3

#N_v is the number of visible units
#N_h is the number of hidden units
#Convention will be that qubits {0,...,N_v} will be visible,
#qubits {N_v+1,...,N} will be hidden
N_v1 = N1

N2 = N1
N_v2 = N2
N_h2 = N2 - N_v2

N3 = N1
N_v3 = N3
N_h3 = N3 - N_v3

#Initialise a 3 dim array for the graph weights, 2 dim array for biases and gamma parameters
J = np.zeros((N1, N1))
b = np.zeros((N1))
gamma_x = np.zeros((N1))
gamma_y = np.zeros((N1))

#Initialize Parameters, J, b for epoch 0 at random, gamma = constant = pi/4
J_i, b_i, g_x_i, g_y_i = NetworkParams(N1, J, b, gamma_x, gamma_y)

#Set learning rate for parameter updates
learning_rate1 = 0.005
#learning_rate2 = 0.001
#learning_rate3 = 0.0001
N_data_samples = 1

N_born_samples1 = 1
N_bornplus_samples1 = 1
N_bornminus_samples1 = 1
N_kernel_samples1 = 1

# N_born_samples2 = 1
# N_bornplus_samples2 = 1
# N_bornminus_samples2 =1
# N_kernel_samples2 = 1
#
# N_born_samples3 = 1
# N_bornplus_samples3 = 1
# N_bornminus_samples3 = 1
# N_kernel_samples3 = 1

kernel_type1 = 'Quantum'
# kernel_type2 = 'Quantum'
# kernel_type3 = 'Quantum'

approx1 = 'Exact'
# approx2 = 'Exact'
# approx3 = 'Exact'

data_samples, data_exact_dict1 = DataImport(approx1, N_v1, N_data_samples)
# data_samples, data_exact_dict2 = DataImport(approx2, N_v2, N_data_samples)
# data_samples, data_exact_dict3 = DataImport(approx3, N_v3, N_data_samples)

L1, J1, b1, gamma_x1, gamma_y1 = PlotMMD(N1, N_v1, N_epochs, J_i, b_i, g_x_i, g_y_i,learning_rate1, approx1, kernel_type1,data_samples, data_exact_dict1,\
                                            N_data_samples, N_born_samples1, N_bornplus_samples1, N_bornminus_samples1, N_kernel_samples1, 'r')
# L2, J2, b2, gamma_x2, gamma_y2 = PlotMMD(N2, N_v2, N_epochs,J_i, b_i, g_x_i, g_y_i,learning_rate2, approx2, kernel_type2, data_samples, data_exact_dict2, \
#                                             N_data_samples, N_born_samples2, N_bornplus_samples2, N_bornminus_samples2, N_kernel_samples2, 'g')
# L3, J3, b3, gamma_x3, gamma_y3 = PlotMMD(N3, N_v3,N_epochs,J_i, b_i, g_x_i, g_y_i, learning_rate3, approx3, kernel_type3, data_samples, data_exact_dict3, \
#                                             N_data_samples, N_born_samples3, N_bornplus_samples3, N_bornminus_samples3, N_kernel_samples3, 'b')

plt.legend(prop={'size': 7}, loc='best').draggable()
# plt.savefig("MMD_%iNv_%s_%s_%.3fLR_%s_%s_%.3fLR_%s_%s_%.3fLR_%iEpochs.pdf" \
#     %(N_v1, kernel_type1[0], approx1[0], learning_rate1, kernel_type2[0], approx2[0], learning_rate2, kernel_type3[0], approx3[0], learning_rate3, N_epochs))
plt.savefig("MMD_%iNv_%s_%s_%.4fLR_%iEpochs.pdf" \
    %(N_v1, kernel_type1[0], approx1[0], learning_rate1, N_epochs))
console_output = sys.stdout
# sys.stdout = open("Output_MMD_%iNv_%s_%iBS_%iNv_%s_%iBS_%iNv_%s_%iBS_%iEpochs_%i_DS" \
#     %(N_v1, kernel_type1[0], N_born_samples1, N_v2, kernel_type2[0], N_born_samples2, N_v3, kernel_type3[0], N_born_samples3,  N_epochs, N_data_samples), 'w')
sys.stdout = open("Output_MMD_%iNv_%s_%s_%.4fLR_%iEpochs" \
    %(N_v1, kernel_type1[0], approx1 ,learning_rate1, N_epochs), 'w')
PrintParamsToFile(J1, b1, L1, N_v1, kernel_type1, N_born_samples1, N_epochs, N_data_samples, learning_rate1)
#PrintParamsToFile(J2, b2, L2, N_v2, kernel_type2, N_born_samples2, N_epochs, N_data_samples, learning_rate2)
#print_params_to_file(J3, b3, L3, N_v3, kernel_type3, N_born_samples3, N_epochs, N_data_samples, learning_rate3)

sys.stdout.close()
sys.stdout= console_output
