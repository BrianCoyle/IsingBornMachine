from pyquil.quil import Program
from pyquil.paulis import *
from pyquil.gates import *
import numpy as np
from pyquil.api import get_qc, WavefunctionSimulator
from random import *

from param_init import HadamardToAll
from auxiliary_functions import IntegerToString

def EncodingFunc(N_qubits, sample):
	'''This function defines  Non-Linear function for encoded samples for Quantum Kernel Cirucit
	to act as graph weights/biases'''

	if (type(sample) is np.ndarray):
		ZZ = np.zeros((N_qubits, N_qubits))
		Z = np.zeros((N_qubits))
		for qubit in range(0, N_qubits):
			Z[qubit] = (np.pi/4)*int(sample[qubit])
			j = 0
			while (j < qubit):
				# if int(sample[qubit]) == 1 and int(sample[j]) == 1:
					# ZZ[qubit,j] = np.pi
					# ZZ[j,qubit] = ZZ[qubit,j]
				ZZ[qubit,j] = (np.pi/4 - int(sample[qubit]))*(np.pi/4 - int(sample[j]))
				ZZ[j,qubit] = ZZ[qubit,j]
				j = j+1
		encoded_sample = {}
		encoded_sample['Interaction'] = ZZ
		encoded_sample['Local'] = Z
	else: raise IOError('\'samples\' must be a numpy array')
	return encoded_sample

def TwoQubitGate(prog, two_q_arg, qubit_1, qubit_2):
	return prog.inst(CPHASE(4*two_q_arg, qubit_1, qubit_2)).inst(PHASE(-2*two_q_arg, qubit_1)).inst(PHASE(-2*two_q_arg, qubit_2))

def IQPLayer(prog, qubits, phi_Z, phi_ZZ):
	N_qubits = len(qubits)
	for j in range(0, N_qubits):
		#Apply local Z rotations (b) to each qubit
		#If the particular qubit sample == 0, apply no gate
		if (phi_Z[j] != False):
			prog.inst(PHASE(-2*phi_Z[j], qubits[j]))
		#Apply Control-Phase(Phi_ZZ_1) gates to each qubit
		for i in range(0, N_qubits): 
			if (i < j):
				#If the particular qubit sample pair == 0, apply no gate
				if (phi_ZZ[i,j] != False):
					prog = TwoQubitGate(prog, phi_ZZ[i,j], qubits[i], qubits[j])
	return prog
	
def KernelCircuit(qc, sample1, sample2):
	'''Compute Quantum kernel given samples from the Born Machine (born_samples) and the Data Distribution (data_samples)
		This must be done for every sample from each distribution (batch gradient descent), (x, y)'''
	'''First layer, sample from first distribution (1), parameters phi_ZZ_1, phi_Z_1'''
	
	qubits = qc.qubits()
	N_qubits = len(qubits)

	prog = Program()

	kernel_circuit_params1 = EncodingFunc(N_qubits, sample1)
	kernel_circuit_params2 = EncodingFunc(N_qubits, sample2)

	phi_ZZ_1 = kernel_circuit_params1['Interaction']
	phi_ZZ_2 = kernel_circuit_params2['Interaction']

	phi_Z_1 = kernel_circuit_params1['Local']
	phi_Z_2 = kernel_circuit_params2['Local']

	###################################################################
	'''First Layer, encoding samples from first distributions, (y)'''
	###################################################################
	'''First layer of Hadamards'''
	prog = HadamardToAll(prog, qubits)
	'''First IQP layer, encoding sample y'''
	prog  = IQPLayer(prog, qubits, phi_Z_1, phi_ZZ_1)

	###################################################################
	'''Second Layer, encoding samples from both distributions, (x, y)'''
	###################################################################
	'''Second layer of Hadamards'''
	prog = HadamardToAll(prog, qubits)
	'''Second IQP layer, encoding samples (x, y)'''
	prog = IQPLayer(prog, qubits, phi_Z_1-phi_Z_2, phi_ZZ_1-phi_ZZ_2)
	
	###################################################################
	'''Third Layer, encoding samples from first distributions, (y)'''
	###################################################################
	'''Third layer of Hadamards'''
	prog = HadamardToAll(prog, qubits)
	'''Second IQP layer, encoding samples (x, y)'''
	prog = IQPLayer(prog, qubits, -phi_Z_2, -phi_ZZ_2) 	#minus sign for complex conjugate

	'''Final layer of Hadamards'''
	prog = HadamardToAll(prog, qubits)

	return prog


def QuantumKernel(qc, N_kernel_samples, sample1, sample2):
	'''This function computes the Quantum kernel for a single pair of samples'''
	

	if type(sample1) is np.ndarray and sample1.ndim != 1: #Check if there is only a single sample in the array of samples
		raise IOError('sample1 must be a 1D numpy array')
	if type(sample2) is np.ndarray and sample2.ndim != 1: #Check if there is only a single sample in the array of samples
		raise IOError('sample2 must be a 1D numpy array')
	
	qubits 		= qc.qubits()
	N_qubits 	= len(qubits)
	make_wf 	= WavefunctionSimulator()


	#run quantum circuit for a single pair of encoded samples
	prog 			= KernelCircuit(qc, sample1, sample2)
	kernel_outcomes = make_wf.wavefunction(prog).get_outcome_probs()

	#Create zero string to read off probability
	zero_string 	= '0'*N_qubits
	kernel_exact 	= kernel_outcomes[zero_string]

	if (N_kernel_samples == 'infinite'):
		#If the kernel is computed exactly, approximate kernel is equal to exact kernel
		kernel_approx = kernel_exact
	else:

		#Index list for classical registers we want to put measurement outcomes into.
		#Measure the kernel circuit to compute the kernel approximately, the kernel is the probability of getting (00...000) outcome.
		#All (N_qubits) qubits are measured at once into dictionary, convert into array
		kernel_measurements_all_qubits_dict = qc.run_and_measure(prog, N_kernel_samples)
		kernel_measurements_used_qubits = np.flip(np.vstack(kernel_measurements_all_qubits_dict[q] for q in sorted(qubits)).T, 1)

		#m is total number of samples, n is the number of used qubits
		(m,n) = kernel_measurements_used_qubits.shape

		N_zero_strings = m - np.count_nonzero(np.count_nonzero(kernel_measurements_used_qubits, 1))
		#The kernel is given by = [Number of times outcome (00...000) occurred]/[Total number of measurement runs]

		kernel_approx = N_zero_strings/N_kernel_samples
	return kernel_exact, kernel_approx

def QuantumKernelArray(qc, N_kernel_samples, samples1, samples2):
	'''This function computes the quantum kernel for all pairs of samples'''
	if type(samples1) is np.ndarray:
		if samples1.ndim == 1: #Check if there is only a single sample in the array of samples
			N_samples1 = 1
		else:
			N_samples1 = samples1.shape[0]
	else: N_samples1 = len(samples1)

	if type(samples2) is np.ndarray:
		if samples2.ndim == 1:
			N_samples2 = 1
		else:
			N_samples2 = samples2.shape[0]
	else: N_samples2 = len(samples2)

	N_qubits = len(qc.qubits())

	kernel_approx_array = np.zeros((N_samples1, N_samples2))
	kernel_exact_array = np.zeros((N_samples1, N_samples2))
	#define a dictionary for both approximate and exact kernel
	kernel_approx_dict = {}
	kernel_exact_dict = {}
	
	for sample_index2 in range(0, N_samples2):
		for sample_index1 in range(0, sample_index2+1):

			s_temp1 = IntegerToString(sample_index1, N_qubits)
			s_temp2 = IntegerToString(sample_index2, N_qubits)
			
			kernel_approx_array[sample_index1, sample_index2], kernel_exact_array[sample_index1, sample_index2],\
			= QuantumKernel(qc, 			\
							N_kernel_samples, 		\
							samples1[sample_index1],\
							samples2[sample_index2]	)
			
			#kernel is symmetric, k(x,y) = k(y,x)
			kernel_approx_array[sample_index2, sample_index1] = kernel_approx_array[sample_index1, sample_index2]
			kernel_exact_array[sample_index2, sample_index1] = kernel_exact_array[sample_index1, sample_index2]

			kernel_approx_dict[s_temp1, s_temp2] = kernel_approx_array[sample_index1, sample_index2]
			kernel_approx_dict[s_temp2, s_temp1] = kernel_approx_dict[s_temp1, s_temp2]

			kernel_exact_dict[s_temp1, s_temp2] = kernel_exact_array[sample_index1, sample_index2]
			kernel_exact_dict[s_temp2, s_temp1] = kernel_exact_dict[s_temp1, s_temp2]

	return kernel_approx_array, kernel_exact_array, kernel_approx_dict, kernel_exact_dict
