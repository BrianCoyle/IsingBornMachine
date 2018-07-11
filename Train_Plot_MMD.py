import numpy as np
import matplotlib.pyplot as plt

from Sample_Gen import born_sampler, plusminus_sample_gen
from MMD_Kernel_Comp import kernel_computation, encoding_func, mmd_cost
from Train_Generation import training_data, data_sampler
from Classical_Kernel import gaussian_kernel
from MMD_Grad_Comp import mmd_train, mmd_grad_kernel_comp, mmd_grad_comp

import time
import sys

t = time.time()

#N_epoch is the total number of training epochs
N_epochs = 7

#N is the total number of qubits.
N1 = 6

#N_v is the number of visible units
#N_h is the number of hidden units
#Convention will be that qubits {0,...,N_v} will be visible,
#qubits {N_v+1,...,N} will be hidden
N_v1 = 6
N_h1 = N1 - N_v1

#Set learning rate for parameter updates
learning_rate = 0.001

N_data_samples = 100

N_born_samples1 = 100
N_bornplus_samples1 = 20
N_bornminus_samples1 = 20
N_kernel_samples1 = 10

kernel_type1 = 'Quantum'


#Define training data along with all binary strings on the visible and hidden variables from train_generation
#M_h is the number of hidden Bernoulli modes in the data
M_h = 5

data = data_sampler(N_v1, N_h1, M_h, N_data_samples)
#print("The Training Data over", N_v1, "Visible qubits is", data1)

#Output MMD Loss function and parameter values, for given number of training samples
L1, J1, b1, gamma_x1, gamma_y1  = mmd_train(N1, N_h1, N_v1, N_epochs, N_data_samples, N_born_samples1,\
                             N_bornplus_samples1,  N_bornminus_samples1, N_kernel_samples1, data, kernel_type1, learning_rate)

sys.stdout = open('MMD_%iNv_%iNepochs' % (N_v1, N_epochs), 'w')
print("\n\nThe data for %i qubits, %i Epochs, with the MMD is given by: \n" % (N_v1, N_epochs))

for epoch in range(0, N_epochs-1):
	print('The weights for Epoch', epoch ,'are :', J1[:,:,epoch], '\n')
	print('The biases for Epoch', epoch ,'are :', b1[:,epoch], '\n')
	print('MMD Loss for Epoch', epoch ,'is:', L1[epoch], '\n')


if kernel_type1 == 'Quantum':
    plt.plot(L1, 'ro', label ='%i Qbs, %i Data Samples,  %i Born Samples for a %s kernel with %i Measurements.' %(N_v1, N_data_samples, N_born_samples1, kernel_type1, N_kernel_samples1))
else:
    plt.plot(L1, 'bo', label ='%i Qbs, %i Data Samples,  %i Born Samples with %s kernel.' %(N_v1, N_data_samples, N_born_samples1, kernel_type1))
#plt.legend('%i V Qbs, %i H Qbs' % (N_v1, N_h1))
plt.xlabel("Epochs")
plt.ylabel("MMD Loss")
plt.title("MMD Loss for %i qubits" % N_v1)

#Time the code execution time
elapsed = time.time() - t

print('Time taken is: ', elapsed)
#############################################################
N2 = 6
N_v2 = 6
N_h2 = N2 - N_v2

N_born_samples2 = 100
N_bornplus_samples2 = 20
N_bornminus_samples2 = 20
N_kernel_samples2 = 50

kernel_type2 = 'Quantum'

L2, J2, b2, gamma_x2, gamma_y2  = mmd_train(N2, N_h2, N_v2, N_epochs, N_data_samples, N_born_samples2,\
                             N_bornplus_samples2, N_bornminus_samples2, N_kernel_samples2, data, kernel_type2, learning_rate)


sys.stdout = open('MMD_%iNv_%iNepochs' % (N_v2, N_epochs), 'a')

print("\n\nThe data for %i qubits, %i Epochs, with the MMD is given by: \n" % (N_v2, N_epochs))

for epoch in range(0, N_epochs - 1):
	print('The weights for Epoch', epoch ,'are :', J2[:,:,epoch], '\n')
	print('The biases for Epoch', epoch ,'are :', b2[:,epoch], '\n')
	print('MMD for Epoch', epoch ,'is:', L2[epoch], '\n')

if kernel_type2 == 'Quantum':
    plt.plot(L2, 'bo', label ='%i Qbs, %i Data Samples,  %i Born Samples from a %s kernel from %i Measurements.' %(N_v2, N_data_samples, N_born_samples2, kernel_type2, N_kernel_samples2))
else:
    plt.plot(L2, 'bo', label ='%i Qbs, %i Data Samples,  %i Born Samples with %s kernel.' %(N_v2, N_data_samples, N_born_samples2, kernel_type2))

plt.legend(prop={'size': 7})
plt.savefig("MMD_%iNv_%s_%iNv_%s_%iEpochs.png" %(N_v1, kernel_type1, N_v2, kernel_type2, N_epochs))
