from pyquil.quil import Program
from pyquil.paulis import *
from pyquil.gates import *
import numpy as np
from pyquil.api import QVMConnection
from pyquil.quilbase import DefGate

from param_init import StateInit, NetworkParams
from train_generation import DataSampler

import sys
qvm = QVMConnection()
p = Program()

'''This Program generates samples from the output distribution of the IQP/QAOA/IQPy circuit according to the Born Rule:
	P(z) = |<z|U|s>|^2, where |s> is the uniform superposition'''
def BornSampler(N, N_v, N_h,
 				N_born_samples,
				J, b, gamma_x, gamma_y):
	#final_layer = ('IQP'for IQP), = ('QAOA' for QAOA), = ('IQPy' for Y-Rot)
	#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, ='NEITHER' for neither
	#sign = 'POSITIVE' to run the positive circuit, = 'NEGATIVE' for the negative circuit, ='NEITHER' for neither
	prog, wavefunction, born_probs_dict = StateInit(N, N_v, N_h, J, b,  gamma_x, gamma_y, 0, 0, 0, 'QAOA', 'NEITHER', 'NEITHER')

	'''Generate (N_samples) samples from output distribution on (N_v) visible qubits'''
	#Index list for classical registers we want to put measurement outcomes into.
	classical_regs = list(range(0, N_v))

	for qubit_index in range(0, N_v):
		prog.measure(qubit_index, qubit_index)

	born_samples = np.asarray(qvm.run(prog, classical_regs, N_born_samples))
	return born_samples, born_probs_dict

def PlusMinusSampleGen(N, N_v, N_h,
 						J, b, gamma_x, gamma_y,
						p, q, r, final_layer, control,
						N_born_samples_plus, N_born_samples_minus):
	''' This function computes the samples required in the estimator, in the +/- terms of the MMD loss function gradient
	 with respect to parameter, J_{p, q} (control = 'WEIGHT')  or b_r (control = 'BIAS')
	'''

	#probs_minus, probs_plus are the exact probabilites outputted from the circuit
	prog_plus, wavefunc_plus, born_probs_dict_plus = StateInit(N, N_v, N_h, J, b,  gamma_x, gamma_y,\
	 												p, q, r, final_layer, control, 'POSITIVE')
	prog_minus, wavefunc_minus, born_probs_dict_minus = StateInit(N, N_v, N_h, J, b,  gamma_x, gamma_y,\
	 														p, q, r, final_layer, control, 'NEGATIVE')


	for qubit_index in range(0, N_v):
		prog_plus.measure(qubit_index, qubit_index)
		prog_minus.measure(qubit_index, qubit_index)
	#Index list for classical registers we want to put measurement outcomes into.
	classical_regs_plus = list(range(0, N_v))
	classical_regs_minus = list(range(0, N_v))
	#generate samples from measurements of shifted circuits
	born_samples_plus = np.asarray(qvm.run(prog_plus, classical_regs_plus, N_born_samples_plus))
	born_samples_minus = np.asarray(qvm.run(prog_minus, classical_regs_minus, N_born_samples_minus))

	return  born_samples_plus, born_samples_minus, born_probs_dict_plus, born_probs_dict_minus
