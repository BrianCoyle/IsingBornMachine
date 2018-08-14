import numpy as np
import matplotlib.pyplot as plt

from train_generation import TrainingData
from param_init import NetworkParams, StateInit
from kl_div import Train
import time
import sys

#Set learning rate for parameter updates
learning_rate = 0.1
#N_epoch is the total number of training epochs
N_epochs = 10
#N is the total number of qubits.

#N_v is the number of visible units
#N_h is the number of hidden units
#Convention will be that qubits {0,...,N_v} will be visible,
#qubits {N_v+1,...,N} will be hidden
N1 = 8
N_v1 = 7
N_h1 = N1 - N_v1
N2 = 9
N_v2 = 7
N_h2 = N2 - N_v2
N3 = 10
N_v3 = 7
N_h3 = N3 - N_v3
#Define training data along with all binary strings on the visible and hidden variables from train_generation
#M_h is the number of hidden Bernoulli modes in the data
M_h = 8

data1, bin_visible1, bin_hidden1 = TrainingData(N_v1, N_h1, M_h)
data2, bin_visible2, bin_hidden2 = TrainingData(N_v2, N_h2, M_h)
data3, bin_visible3, bin_hidden3 = TrainingData(N_v3, N_h3, M_h)

print("The Training Data over", N_v1, "Visible qubits is", data1)
print("Data Summed is: ", data1.sum())

# #Output KL Divergence, Output Probability Distribution, and all parameter values for all epochs
KL1, P_v1, J1, b1, gamma_x1, gamma_y1 =  KLTrain(N1, N_h1, N_v1, N_epochs, data1, bin_visible1, bin_hidden1, learning_rate)
KL2, P_v2, J2, b2, gamma_x2, gamma_y2 =  KLTrain(N2, N_h2, N_v2, N_epochs, data1, bin_visible2, bin_hidden2, learning_rate)
KL3, P_v3, J3, b3, gamma_x3, gamma_y3 =  KLTrain(N3, N_h3, N_v3, N_epochs, data1, bin_visible3, bin_hidden3, learning_rate)


console_output = sys.stdout
sys.stdout = open('KL_%iNv_%iNepochs' % (N_v1,N_epochs), 'w')
print("The data for %i qubits, %i Visible/%i Hidden, %i Epochs is given by: \n" % (N1, N_v1 ,N_h1, N_epochs))

for epoch in range(0, N_epochs-1):
	print('The weights for Epoch', epoch ,'are :', J1[:,:,epoch], '\n')
	print('The biases for Epoch', epoch ,'are :', b1[:,epoch], '\n')
	print('KL1 Divergence for EPOCH', epoch ,'is:', KL1[epoch], '\n')
	print('P_v1 for epoch ', epoch,' is ', P_v1[:,epoch],'\n\n')

plt.plot(KL1, 'ro-', label='%i Visible Qbs, %i Hidden Qbs' %(N_v1, N_h1))

plt.xlabel("Epochs")#=
plt.ylabel("KL Divergence")
plt.title("KL Divergence for %i visible qubits" % N_v1)

#Time the code execution time
elapsed1 = time.time() - t1

print('Time taken is: ', elapsed1)

sys.stdout.close()
sys.stdout= console_output
#############################################################

console_output = sys.stdout
sys.stdout = open('KL_%iNv_%iNepochs' % (N_v1,N_epochs), 'a')
print("\n\nThe data for %i qubits, %i Visible/%i Hidden, %i Epochs is given by" % (N2, N_v2 ,N_h2, N_epochs))

for epoch in range(0, N_epochs - 1):
	print('The weights for Epoch', epoch ,'are :', J2[:,:,epoch], '\n')
	print('The biases for Epoch', epoch ,'are :', b2[:,epoch], '\n')
	print('KL2 Divergence for Epoch', epoch ,'is:', KL2[epoch], '\n')
	print('P_v2 for epoch ', epoch,' is ', P_v2[:,epoch], '\n\n')


plt.plot(KL2, 'bo-', label ='%i Visible Qbs, %i Hidden Qbs' %(N_v2, N_h2))


sys.stdout.close()
sys.stdout= console_output
#################################################################

console_output = sys.stdout
sys.stdout = open('KL_%iNv_%iNepochs_2' % (N_v1,N_epochs), 'a')
print("\n\nThe data for %i qubits, %i Visible/%i Hidden, %i Epochs is given by" % (N3, N_v3 ,N_h3, N_epochs))

for epoch in range(0, N_epochs - 1):
	print('The weights for Epoch', epoch ,'are :', J3[:,:,epoch], '\n')
	print('The biases for Epoch', epoch ,'are :', b3[:,epoch], '\n')
	print('KL2 Divergence for Epoch', epoch ,'is:', KL3[epoch], '\n')
	print('P_v2 for epoch ', epoch,' is ', P_v3[:,epoch], '\n\n')


plt.plot(KL3, 'go-', label ='%i Visible Qbs, %i Hidden Qbs' %(N_v3, N_h3))
plt.legend()

sys.stdout.close()
sys.stdout= console_output
#plt.savefig("KL_%iNv_%iNh_%iNv_%iNh_%iNv_%i_Nh.png" %( N_v1, N_h1, N_v2, N_h2, N_v3, N_h3))

plt.savefig("KL_%iNv_%iNh_%iNv_%i_Nh.pdf" %(N_v2, N_h2, N_v3, N_h3))
