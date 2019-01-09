from pyquil.quil import Program
from pyquil.paulis import *
from pyquil.gates import *
import numpy as np
from pyquil.api import get_qc, WavefunctionSimulator
from pyquil.quilbase import DefGate

from param_init import StateInit, NetworkParams
from train_generation import DataSampler
from auxiliary_functions import EmpiricalDist
import sys

'''This Program generates samples from the output distribution of the IQP/QAOA/IQPy circuit according to the Born Rule:
	P(z) = |<z|U|s>|^2, where |s> is the uniform superposition'''
def BornSampler(device_params, N_samples, circuit_params, circuit_choice):

	device_name = device_params[0]
	as_qvm_value = device_params[1]
	make_wf = WavefunctionSimulator()

	qc = get_qc(device_name, as_qvm = as_qvm_value)
	#final_layer = ('IQP'for IQP), = ('QAOA' for QAOA), = ('IQPy' for Y-Rot)
	#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, ='GAMMA' for gamma params, ='NEITHER' for neither
	#sign = 'POSITIVE' to run the positive circuit, = 'NEGATIVE' for the negative circuit, ='NEITHER' for neither
	prog = StateInit(device_params, circuit_params, 0, 0, 0, 0, circuit_choice , 'NEITHER', 'NEITHER')

	# wavefunction = make_wf.wavefunction(prog)
	# born_probs_dict = wavefunction.get_outcome_probs()
	# print(control, sign, outcome_dict, prog)
	N_born_samples = N_samples[1]
	'''Generate (N_born_samples) samples from output distribution on (N_qubits) visible qubits'''
	born_samples_all_qubits_dict = qc.run_and_measure(prog, N_born_samples)
	#All (5) qubits are measured at once
	born_samples = np.flip(np.vstack(born_samples_all_qubits_dict[q] for q in sorted(qc.qubits())).T, 1)

	born_probs_approx_dict = EmpiricalDist(born_samples, len(qc.qubits())) #Compute empirical distribution of the output samples
	born_probs_exact_dict = make_wf.wavefunction(prog).get_outcome_probs()

	return born_samples, born_probs_approx_dict, born_probs_exact_dict

def PlusMinusSampleGen(device_params, circuit_params,
						p, q, r, s, circuit_choice, control, N_samples):
	''' This function computes the samples required in the estimator, in the +/- terms of the MMD loss function gradient
	 with respect to parameter, J_{p, q} (control = 'WEIGHT') , b_r (control = 'BIAS') or gamma_x_s (control == 'GAMMA')
	'''	
	make_wf = WavefunctionSimulator()

	device_name = device_params[0]
	as_qvm_value = device_params[1]

	qc = get_qc(device_name, as_qvm = as_qvm_value)

	#probs_minus, probs_plus are the exact probabilites outputted from the circuit
	prog_plus= StateInit(device_params, circuit_params, p, q, r, s, circuit_choice, control, 'POSITIVE')
	prog_minus = StateInit(device_params, circuit_params, p, q, r, s, circuit_choice, control, 'NEGATIVE')
	# wavefunction_plus = make_wf.wavefunction(prog_plus)
	# born_probs_dict_plus = wavefunction_plus.get_outcome_probs()
	# wavefunction_minus = make_wf.wavefunction(prog_minus)
	# born_probs_dict_minus = wavefunction_minus.get_outcome_probs()

	batch_size = N_samples[2]
	#generate batch_size samples from measurements of + shifted circuits 
	born_samples_plus_all_qbs_dict = qc.run_and_measure(prog_plus, batch_size)
	#All (5) qubits are measured at once
	born_samples_plus = np.flip(np.vstack(born_samples_plus_all_qbs_dict[q] for q in sorted(qc.qubits())).T, 1)

	#generate batch_size samples from measurements of - shifted circuits 
	born_samples_minus_all_qbs_dict = qc.run_and_measure(prog_minus, batch_size)
	#All (5) qubits are measured at once in QPU (and QVM in run_and_measure)
	born_samples_minus = np.flip(np.vstack(born_samples_minus_all_qbs_dict[q] for q in sorted(qc.qubits())).T, 1)

	return  born_samples_plus, born_samples_minus
	# , born_probs_dict_plus, born_probs_dict_minus


