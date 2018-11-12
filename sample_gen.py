from pyquil.quil import Program
from pyquil.paulis import *
from pyquil.gates import *
import numpy as np
from pyquil.api import get_qc, WavefunctionSimulator
from pyquil.quilbase import DefGate

from param_init import StateInit, NetworkParams
from train_generation import DataSampler

import sys
qc = get_qc("9q-generic-qvm")
prog = Program()

'''This Program generates samples from the output distribution of the IQP/QAOA/IQPy circuit according to the Born Rule:
	P(z) = |<z|U|s>|^2, where |s> is the uniform superposition'''
def BornSampler(N, N_v, N_h,
 				N_born_samples,
				J, b, gamma_x, gamma_y, circuit_choice):

	#final_layer = ('IQP'for IQP), = ('QAOA' for QAOA), = ('IQPy' for Y-Rot)
	#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, ='GAMMA' for gamma params, ='NEITHER' for neither
	#sign = 'POSITIVE' to run the positive circuit, = 'NEGATIVE' for the negative circuit, ='NEITHER' for neither
	prog, wavefunction, born_probs_dict = StateInit(N, N_v, N_h, J, b,  gamma_x, gamma_y, 0, 0, 0, 0, circuit_choice , 'NEITHER', 'NEITHER')
	# print('Wavefunciton is:\n',wavefunction, '\nBorn probs are:\n', born_probs_dict)
	'''Generate (N_born_samples) samples from output distribution on (N_v) visible qubits'''

	born_samples_all_qubits = np.asarray(qc.run_and_measure(prog, N_born_samples))
	born_samples = np.flip(np.delete(born_samples_all_qubits, range(N_v, 9), axis=1), 1)

	# born_samples = np.flip(born_samples, 1)
	return born_samples, born_probs_dict

def PlusMinusSampleGen(N, N_v, N_h,
 						J, b, gamma_x, gamma_y,
						p, q, r, s, circuit_choice, control,
						N_born_samples_plus, N_born_samples_minus):
	''' This function computes the samples required in the estimator, in the +/- terms of the MMD loss function gradient
	 with respect to parameter, J_{p, q} (control = 'WEIGHT') , b_r (control = 'BIAS') or gamma_x_s (control == 'GAMMA')
	'''

	#probs_minus, probs_plus are the exact probabilites outputted from the circuit
	prog_plus, wavefunc_plus, born_probs_dict_plus = StateInit(N, N_v, N_h, J, b,  gamma_x, gamma_y,\
	 												p, q, r, s, circuit_choice, control, 'POSITIVE')
	prog_minus, wavefunc_minus, born_probs_dict_minus = StateInit(N, N_v, N_h, J, b,  gamma_x, gamma_y,\
	 														p, q, r, s, circuit_choice, control, 'NEGATIVE')

	#generate samples from measurements of shifted circuits
	born_samples_plus_all_qbs = np.asarray(qc.run_and_measure(prog_plus, N_born_samples_plus))
	born_samples_plus = np.flip(np.delete(born_samples_plus_all_qbs, range(N_v, 9), axis=1), 1)

	born_samples_minus_all_qbs = np.asarray(qc.run_and_measure(prog_minus, N_born_samples_minus))
	born_samples_minus = np.flip(np.delete(born_samples_minus_all_qbs, range(N_v, 9), axis=1), 1)

	# # # Flip ordering of samples to be consistent with Rigetti convention
	# born_samples_plus = np.flip(born_samples_plus, 1)
	# born_samples_minus = np.flip(born_samples_minus, 1)

	return  born_samples_plus, born_samples_minus, born_probs_dict_plus, born_probs_dict_minus
