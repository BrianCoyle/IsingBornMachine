from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
import numpy as np
from numpy import pi,log2
from pyquil.api import QVMConnection
from random import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin

import Amplitude_Computation_Hadamard_Test as amp

'''This function computes the amplitude term distribution'''	
def compute_amp_term(N, N_v, N_h, J, b, gamma, p, q, r, bin_visible, bin_hidden, control):

	Amp1 = np.zeros((2**N_v, 2**N_h), dtype = complex)
	Amp2 = np.zeros((2**N_v, 2**N_h), dtype = complex)

	#Define empty arrays, with each index, [string][h] corresponding to one possible string of 
	#one string of the visibles, with one string of the hiddens, (2**N_v)*(2**N_h) possible strings
	
	Probs_Amplitudes  = np.zeros((2**N_v, 2**N_h))
	
	for v_string in range(0, 2**N_v):
		for h_string in range(0, 2**N_h):

			#Compute the two amplitudes for each string of visible and hidden variables
			Amp1[v_string, h_string] =  amp.Amplitude_Computation1(N, N_v, N_h, J, b, gamma, bin_visible[v_string,:], bin_hidden[h_string,:])	
													
			Amp2[v_string, h_string] =  amp.Amplitude_Computation2(N, N_v, N_h, J, b, gamma, p, q, r,  bin_visible[v_string, :], bin_hidden[h_string, :], control)
			
			#print('\n The First Amplitude for epoch is ', bin_visible[v_string, :], bin_hidden[h_string, :],
			#' is: \n', Amp1[v_string, h_string], '\n')
			
	#The update value of the weights for outcomes: v on visibles, h on hiddens
	Probs_Amplitudes = 2*np.multiply(Amp1, Amp2).imag

	#The distribution of the model over the visibles 	
	P_v_Amp = Probs_Amplitudes.sum(axis = 1)
	
	return P_v_Amp


