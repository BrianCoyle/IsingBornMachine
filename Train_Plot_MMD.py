from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
import numpy as np
from numpy import pi
from numpy import log2
from pyquil.api import QVMConnection
from random import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin
import matplotlib.pyplot as plt

from Sample_Gen import born_sampler, plusminus_sample_gen
from MMD_Kernel_Comp import kernel_computation, encoding_func, mmd_cost
from Train_Generation import training_data, data_sampler
from Classical_Kernel import gaussian_kernel
from MMD_Grad_Comp import mmd_train, mmd_grad_kernel_comp, mmd_grad_comp

import time
import sys

qvm = QVMConnection()
p = Program()

t = time.time()

#N_epoch is the total number of training epochs
N_epochs = 100

#N is the total number of qubits.
N1 = 9

#N_v is the number of visible units
#N_h is the number of hidden units
#Convention will be that qubits {0,...,N_v} will be visible, 
#qubits {N_v+1,...,N} will be hidden
N_v1 = 6
N_h1 = N1 - N_v1

N_data_samples1 = 10
N_born_samples1 = 10
N_bornplus_samples1 = 10
N_bornminus_samples1 = 10
N_kernel_samples1 = 5

#Define training data along with all binary strings on the visible and hidden variables from train_generation
#M_h is the number of hidden Bernoulli modes in the data
M_h = 5

data1 = data_sampler(N_v1, N_h1, M_h, N_data_samples1)
print("The Training Data over", N_v1, "Visible qubits is", data1)

#Output MMD Loss function and parameter values, for given number of training samples
L1, J1, b1, gamma_x1, gamma_y1  = mmd_train(N1, N_h1, N_v1, N_epochs, N_data_samples1, N_born_samples1,\
                             N_bornplus_samples1,  N_bornminus_samples1, N_kernel_samples1, data1, 'Gaussian')

sys.stdout = open('%iNv_%iNepochs' % (N_v1,N_epochs), 'w')
print("The data for %i qubits, %i Visible/%i Hidden, %i Epochs is given by: \n" % (N1, N_v1 ,N_h1, N_epochs))

for epoch in range(0, N_epochs-1):
	print('The weights for Epoch', epoch ,'are :', J1[:,:,epoch], '\n')
	print('The biases for Epoch', epoch ,'are :', b1[:,epoch], '\n')
	print('MMD Loss for EPOCH', epoch ,'is:', L1[epoch], '\n')


#epochs = np.arange(N_epochs-1)

plt.plot(L1, 'ro', label='%i Visible Qbs, %i Hidden Qbs' %(N_v1, N_h1))

#plt.legend('%i V Qbs, %i H Qbs' % (N_v1, N_h1))
plt.xlabel("Epochs")#=
plt.ylabel("MMD Loss") 
plt.title("MMD Loss for %i visible qubits" % N_v1)

#Time the code execution time
elapsed = time.time() - t

print('Time taken is: ', elapsed)
#############################################################
# N2 = 10
# N_v2 = 6
# N_h2 = N2 - N_v2

# N_data_samples2 = 10
# N_born_samples2 = 10
# N_bornplus_samples2 = 10
# N_bornminus_samples2 = 10
# N_kernel_samples2 = 5

# L2, J2, b2, gamma_x2, gamma_y2  = mmd_train(N2, N_h2, N_v2, N_epochs, N_data_samples2, N_born_samples2,\
#                              N_bornplus_samples2, N_bornminus_samples2, N_kernel_samples2, data1, 'Quantum Kernel')
# print("L is: " , L2)

# sys.stdout = open('%iNv_%iNepochs' % (N_v2, N_epochs), 'a')

# print("\n\nThe data for %i qubits, %i Visible/%i Hidden, %i Epochs is given by" % (N2, N_v2 ,N_h2, N_epochs))

# for epoch in range(0, N_epochs - 1):
# 	print('The weights for Epoch', epoch ,'are :', J2[:,:,epoch], '\n')
# 	print('The biases for Epoch', epoch ,'are :', b2[:,epoch], '\n')
# 	print('KL2 Divergence for EPOCH', epoch ,'is:', L2[epoch], '\n')


# plt.plot(L2, 'bo', label ='%i Visible Qbs, %i Hidden Qbs' %(N_v2, N_h2))
# plt.legend()
# plt.savefig("MMD_%iN_%iNv_%iNh_%iN_%iNv_%iNh.png" %(N1, N_v1, N_h1, N2, N_v2, N_h2))

# #To Do
# #Different Training Data Generated on each of the two runs
# #Fix data generation to allow for 0 hidden qubits

