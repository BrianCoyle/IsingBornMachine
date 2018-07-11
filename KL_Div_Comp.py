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

from Param_Init import network_params, state_init
from KL_Div_Gradient_Comp import compute_amp_term
from Distribution_Computation import compute_dist

import sys
qvm = QVMConnection()
p = Program()

def train(N, N_h, N_v, N_epochs, data, bin_visible, bin_hidden):

	#Initialise a 3 dim array for the graph weights, 2 dim array for biases and gamma parameters

	J = np.zeros((N, N, N_epochs))
	b = np.zeros((N,  N_epochs))
	gamma_x = np.zeros((N,  N_epochs))
	gamma_y = np.zeros((N,  N_epochs))

	#Set learning rate for parameter updates
	learning_rate = 0.1

	#Initialize Parameters, J, b for epoch 0 at random, gamma = constant = pi/4
	J[:,:,0], b[:,0], gamma_x[:,0], gamma_y[:,0] = network_params(N, J[:,:,0], b[:,0], gamma_x[:,0], gamma_y[:,0])

	#gamma_x/ gamma_y is not to be trained, set gamma values to be constant at each epoch
	for epoch in range(0, N_epochs-1):
		gamma_x[:,epoch + 1] = gamma_x[:, epoch]
		gamma_y[:,epoch + 1] = gamma_y[:, epoch]


	#Initialise the probability distribution for each bias term.
	P_v_Amp_bias = np.zeros((2**N_v))
	update_bias = np.zeros((2**N_v, N))
	delta_bias = np.zeros((N))

	P_v_Amp_weight = np.zeros((2**N_v, N, N))
	update_weight = np.zeros((2**N_v, N, N))
	delta_weight = np.zeros((N, N))

	P_v = np.zeros((2**N_v ,N_epochs))
	KL = np.zeros((N_epochs-1))

	for epoch in range(0, N_epochs-1):

		#Compute output distribution for current set of parameters, J, b, gamma_x
		P_v[:, epoch] = compute_dist(N, N_v, N_h, J[:,:,epoch], b[:,epoch], gamma_x[:, epoch], bin_visible, bin_hidden)

		print('P_v for epoch', epoch, 'is: \n', P_v[:,epoch])
		'''Updating bias b[r], control set to 'BIAS' '''
		for bias_index in range(0,N):
			P_v_Amp_bias \
			= compute_amp_term(N, N_v, N_h, J[:,:,epoch], b[:,epoch], gamma_x[:, epoch], 0, 0, bias_index, bin_visible, bin_hidden, 'BIAS')
			update_bias[:, bias_index] = np.multiply(np.divide(data, P_v[:, epoch]), P_v_Amp_bias)

			delta_bias[bias_index] = update_bias[:, bias_index].sum(axis = 0)

		#print('delta_bias for epoch',  epoch, 'is: ', delta_bias)

		b[:, epoch + 1] = b[:, epoch] - learning_rate*delta_bias


		'''Updating weight J[p,q], control set to 'WEIGHTS' '''
		for j in range(0, N):
			i = 0
			while(i < j):
				P_v_Amp_weight[:, i, j] \
				= compute_amp_term(N, N_v, N_h, J[:,:, epoch], b[:,epoch], gamma_x[:, epoch], i, j, 0, bin_visible, bin_hidden, 'WEIGHTS')

				update_weight[:, i, j] = np.multiply(np.divide(data, P_v[:, epoch]), P_v_Amp_weight[:, i, j])

				delta_weight[i,j] = update_weight[:, i, j].sum(axis = 0)
				#print(delta_weight)
				i = i+1
		J[:,:,epoch+1] = J[:,:, epoch] - learning_rate*(delta_weight + np.transpose(delta_weight))


		#wavefunction, probs = state_init(N, N_v, N_h, J[:,:,epoch], b[:,:,epoch],gamma[:,:,epoch])
		#P_v[:, epoch] = compute_dist(N, N_v, N_h, J[:,:, epoch], b[:,epoch], gamma[:, epoch], i, j, 0, bin_visible, bin_hidden, 1)
		'''Check KL Divergence of model distribution'''

		KL[epoch] = np.multiply(data, log2(np.divide(data, P_v[:, epoch]))).sum()

	return KL, P_v, J, b, gamma_x, gamma_y
