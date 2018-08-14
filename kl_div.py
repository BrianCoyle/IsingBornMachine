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

from param_init import NetworkParams, StateInit

import sys
'''This function computes the amplitude term distribution'''
def ComputeAmpTerm(N, N_v, N_h,
					J, b, gamma,
					p, q, r,
					bin_visible, bin_hidden,
					control):

	amp1 = np.zeros((2**N_v, 2**N_h), dtype = complex)
	amp2 = np.zeros((2**N_v, 2**N_h), dtype = complex)

	#Define empty arrays, with each index, [string][h] corresponding to one possible string of
	#one string of the visibles, with one string of the hiddens, (2**N_v)*(2**N_h) possible strings

	Probs_Amplitudes  = np.zeros((2**N_v, 2**N_h))

	for v_string in range(0, 2**N_v):
		for h_string in range(0, 2**N_h):

			#Compute the two amplitudes for each possible string over the visible and hidden qubits
			amp1[v_string, h_string] =  amp.AmplitudeComputation1(N, N_v, N_h,\
																	J, b, gamma,\
			 														bin_visible[v_string,:], bin_hidden[h_string,:])

			amp2[v_string, h_string] =  amp.AmplitudeComputation2(N, N_v, N_h,
																J, b, gamma, \
																p, q, r, \
																bin_visible[v_string, :], bin_hidden[h_string, :],\
																control)

	#The update value of the weights for outcomes: v on visibles, h on hiddens
	probs_amplitudes = 2*np.multiply(amp1, amp2).imag

	#The distribution of the model over the visibles
	p_v_amp = probs_amplitudes.sum(axis = 1)

	return p_v_Amp


'''This function computes the output distribution after a given epoch to compute the KL Divergence'''
def ComputeDist(N, N_v, N_h, J, b, gamma, bin_visible, bin_hidden):

	output_amplitude = np.zeros((2**N_v, 2**N_h), dtype = complex)

	#Define empty arrays, with each index, [string][h] corresponding to one possible string of
	#a string on the visibles, with one string of the hiddens, there are (2**N_v)*(2**N_h) possible binary strings
	probs_dist = np.zeros((2**N_v, 2**N_h))

	for v_string in range(0, 2**N_v):
		for h_string in range(0, 2**N_h):

			#Compute the amplitudes for each string of visible and hidden variables (with no gradient)
			output_amplitude[v_string, h_string] =  amp.AmplitudeComputation1(N, N_v, N_h, J, b, \
													gamma, bin_visible[v_string,:], bin_hidden[h_string, :])

	probs_dist = np.multiply(output_amplitude, output_amplitude.conjugate()).real
	#The distribution of the model over the visibles variables only
	p_v = probs_dist.sum(axis = 1)

	return  p_v

qvm = QVMConnection()
p = Program()

def KlTrain(N, N_h, N_v, N_epochs, data, bin_visible, bin_hidden, learning_rate):

	#Initialise a 3 dim array for the graph weights, 2 dim array for biases and gamma parameters
	J = np.zeros((N, N, N_epochs))
	b = np.zeros((N,  N_epochs))
	gamma_x = np.zeros((N,  N_epochs))
	gamma_y = np.zeros((N,  N_epochs))

	#Initialize Parameters, J, b for epoch 0 at random, gamma = constant = pi/4
	J[:,:,0], b[:,0], gamma_x[:,0], gamma_y[:,0] = NetworkParams(N, J[:,:,0], b[:,0], gamma_x[:,0], gamma_y[:,0])

	#gamma_x/ gamma_y is not to be trained, set gamma values to be constant at each epoch
	for epoch in range(0, N_epochs-1):
		gamma_x[:,epoch + 1] = gamma_x[:, epoch]
		gamma_y[:,epoch + 1] = gamma_y[:, epoch]


	#Initialise the probability distribution for each bias term.
	p_v_amp_bias = np.zeros((2**N_v))
	update_bias = np.zeros((2**N_v, N))
	delta_bias = np.zeros((N))

	p_v_amp_weight = np.zeros((2**N_v, N, N))
	update_weight = np.zeros((2**N_v, N, N))
	delta_weight = np.zeros((N, N))

	p_v = np.zeros((2**N_v ,N_epochs))
	KL = np.zeros((N_epochs-1))

	for epoch in range(0, N_epochs-1):

		#Compute output distribution for current set of parameters, J, b, gamma_x
		p_v[:, epoch] = ComputeDist(N, N_v, N_h, J[:,:,epoch], b[:,epoch], gamma_x[:, epoch], bin_visible, bin_hidden)

		print('P_v for epoch', epoch, 'is: \n', P_v[:,epoch])
		'''Updating bias b[r], control set to 'BIAS' '''
		for bias_index in range(0,N):
			P_v_Amp_bias \
			= ComputeAmpTerm(N, N_v, N_h, J[:,:,epoch], b[:,epoch], gamma_x[:, epoch], 0, 0, bias_index, bin_visible, bin_hidden, 'BIAS')
			update_bias[:, bias_index] = np.multiply(np.divide(data, P_v[:, epoch]), P_v_Amp_bias)

			delta_bias[bias_index] = update_bias[:, bias_index].sum(axis = 0)

		b[:, epoch + 1] = b[:, epoch] - learning_rate*delta_bias

		'''Updating weight J[p,q], control set to 'WEIGHTS' '''
		for j in range(0, N):
			i = 0
			while(i < j):
				p_v_amp_weight[:, i, j] \
				= ComputeAmpTerm(N, N_v, N_h, J[:,:, epoch], b[:,epoch], gamma_x[:, epoch], i, j, 0, bin_visible, bin_hidden, 'WEIGHTS')

				update_weight[:, i, j] = np.multiply(np.divide(data, p_v[:, epoch]), p_v_amp_weight[:, i, j])

				delta_weight[i,j] = update_weight[:, i, j].sum(axis = 0)
				#print(delta_weight)
				i = i+1
		J[:,:,epoch+1] = J[:,:, epoch] - learning_rate*(delta_weight + np.transpose(delta_weight))


		#wavefunction, probs = state_init(N, N_v, N_h, J[:,:,epoch], b[:,:,epoch],gamma[:,:,epoch])
		#P_v[:, epoch] = compute_dist(N, N_v, N_h, J[:,:, epoch], b[:,epoch], gamma[:, epoch], i, j, 0, bin_visible, bin_hidden, 1)
		'''Check KL Divergence of model distribution'''
		KL[epoch] = np.multiply(data, log2(np.divide(data, p_v[:, epoch]))).sum()

	return KL, P_v, J, b, gamma_x, gamma_y
