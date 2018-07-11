import numpy as np
import matplotlib.pyplot as plt

from Train_Generation import training_data
from Param_Init import network_params, state_init
from KL_Div_Comp import train
import time
import sys


t = time.time()

#N_epoch is the total number of training epochs
N_epochs = 5

#N is the total number of qubits.
N1 = 3

#N_v is the number of visible units
#N_h is the number of hidden units
#Convention will be that qubits {0,...,N_v} will be visible,
#qubits {N_v+1,...,N} will be hidden
N_v1 = 2
N_h1 = N1 - N_v1


#Define training data along with all binary strings on the visible and hidden variables from train_generation
#M_h is the number of hidden Bernoulli modes in the data
M_h = 5

data1, bin_visible1, bin_hidden1 = training_data(N_v1, N_h1, M_h)
print("The Training Data over", N_v1, "Visible qubits is", data1)
print("Data Summed is: ", data1.sum())

#Output KL Divergence, Output Probability Distribution, and all parameter values for all epochs
KL1, P_v1, J, b, gamma_x, gamma_y =  train(N1, N_h1, N_v1, N_epochs, data1, bin_visible1, bin_hidden1)

sys.stdout = open('KL_%iNv_%iNepochs' % (N_v1,N_epochs), 'w')
print("The data for %i qubits, %i Visible/%i Hidden, %i Epochs is given by: \n" % (N1, N_v1 ,N_h1, N_epochs))

for epoch in range(0, N_epochs-1):
	print('The weights for Epoch', epoch ,'are :', J[:,:,epoch], '\n')
	print('The biases for Epoch', epoch ,'are :', b[:,epoch], '\n')
	print('KL1 Divergence for EPOCH', epoch ,'is:', KL1[epoch], '\n')
	print('P_v1 for epoch ', epoch,' is ', P_v1[:,epoch],'\n\n')

plt.plot(KL1, 'ro', label='%i Visible Qbs, %i Hidden Qbs' %(N_v1, N_h1))

plt.xlabel("Epochs")#=
plt.ylabel("KL Divergence")
plt.title("KL Divergence for %i visible qubits" % N_v1)

#Time the code execution time
elapsed = time.time() - t

print('Time taken is: ', elapsed)
#############################################################
N2 = 4
N_v2 = 2
N_h2 = N2 - N_v2

data2, bin_visible2, bin_hidden2 = training_data(N_v2, N_h2, M_h)

KL2, P_v2, J, b, gamma_x, gamma_y =  train(N2, N_h2, N_v2, N_epochs, data1, bin_visible2, bin_hidden2)

sys.stdout = open('KL_%iNv_%iNepochs' % (N_v1,N_epochs), 'a')
print("\n\nThe data for %i qubits, %i Visible/%i Hidden, %i Epochs is given by" % (N2, N_v2 ,N_h2, N_epochs))

for epoch in range(0, N_epochs - 1):
	print('The weights for Epoch', epoch ,'are :', J[:,:,epoch], '\n')
	print('The biases for Epoch', epoch ,'are :', b[:,epoch], '\n')
	print('KL2 Divergence for EPOCH', epoch ,'is:', KL2[epoch], '\n')
	print('P_v2 for epoch ', epoch,' is ', P_v2[:,epoch], '\n\n')


plt.plot(KL2, 'bo', label ='%i Visible Qbs, %i Hidden Qbs' %(N_v2, N_h2))
plt.legend()

plt.savefig("KL_%iN_%iNv_%iNh_%iN_%iNv_%iNh.png" %(N1, N_v1, N_h1, N2, N_v2, N_h2))
