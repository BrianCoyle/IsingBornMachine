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


def BornSampler(qc, N_samples, circuit_params, circuit_choice):
	'''
	This Program generates samples from the output distribution of the IQP/QAOA/IQPy circuit according to the Born Rule:
	P(z) = |<z|U|s>|^2, where |s> is the uniform superposition
	'''
	
	make_wf = WavefunctionSimulator()

	N_qubits = len(qc.qubits())
	#final_layer = ('IQP'for IQP), = ('QAOA' for QAOA), = ('IQPy' for Y-Rot)
	#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, ='GAMMA' for gamma params, ='NEITHER' for neither
	#sign = 'POSITIVE' to run the positive circuit, = 'NEGATIVE' for the negative circuit, ='NEITHER' for neither
	prog = StateInit(qc, circuit_params, 0, 0, 0, 0, circuit_choice , 'NEITHER', 'NEITHER')

	# wavefunction = make_wf.wavefunction(prog)
	# born_probs_dict = wavefunction.get_outcome_probs()
	N_born_samples = N_samples[1]
	'''Generate (N_born_samples) samples from output distribution on (N_qubits) visible qubits'''
	born_samples_all_qubits_dict = qc.run_and_measure(prog, N_born_samples)
	
	born_samples = np.flip(np.vstack(born_samples_all_qubits_dict[q] for q in sorted(qc.qubits())).T, 1) #put outcomes into array

	born_probs_approx_dict = EmpiricalDist(born_samples, N_qubits, 'full_dist') #Compute empirical distribution of the output samples
	born_probs_exact_dict = make_wf.wavefunction(prog).get_outcome_probs()

	return born_samples, born_probs_approx_dict, born_probs_exact_dict

def PlusMinusSampleGen(qc, circuit_params,
						p, q, r, s, circuit_choice, control, N_samples):
	''' This function computes the samples required in the estimator, in the +/- terms of the MMD loss function gradient
	 with respect to parameter, J_{p, q} (control = 'WEIGHT') , b_r (control = 'BIAS') or gamma_x_s (control == 'GAMMA')
	'''	
	
	#probs_minus, probs_plus are the exact probabilites outputted from the circuit
	prog_plus= StateInit(qc, circuit_params, p, q, r, s, circuit_choice, control, 'POSITIVE')
	prog_minus = StateInit(qc, circuit_params, p, q, r, s, circuit_choice, control, 'NEGATIVE')

	batch_size = N_samples[2]
	#generate batch_size samples from measurements of +/- shifted circuits 
	born_samples_plus_all_qbs_dict = qc.run_and_measure(prog_plus, batch_size)
	born_samples_minus_all_qbs_dict = qc.run_and_measure(prog_minus, batch_size)
	born_samples_pm = []
 	#put outcomes into list of arrays
	born_samples_pm.append(np.flip(np.vstack(born_samples_plus_all_qbs_dict[q] for q in sorted(qc.qubits())).T, 1))
	born_samples_pm.append(np.flip(np.vstack(born_samples_minus_all_qbs_dict[q] for q in sorted(qc.qubits())).T, 1))

	return  born_samples_pm


