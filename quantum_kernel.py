from pyquil.quil import Program
from pyquil.paulis import *
from pyquil.gates import *
import numpy as np
from pyquil.api import get_qc, WavefunctionSimulator
from random import *

from param_init import HadamardToAll
from auxiliary_functions import ConvertToString

'''Define Non-Linear function to encode samples in graph weights/biases'''
#Fill graph weight and bias arrays according to one single sample
def EncodingFunc(N_qubits, samples):
	if (type(samples) is np.ndarray):
		N_samples = samples.shape[0]
		ZZ = np.zeros((N_qubits, N_qubits, N_samples))
		Z = np.zeros((N_qubits, N_samples))
		for sample_number in range(0, N_samples):
			for i in range(0, N_qubits):
				if int(samples[sample_number, i]) == 1:
					Z[i, sample_number] = (np.pi)/4
				j = 0
				while (j < i):
					if int(samples[sample_number, i]) == 1 and int(samples[sample_number, j]) == 1:
						ZZ[i,j, sample_number] = (np.pi)/4
						ZZ[j,i, sample_number] = ZZ[i,j, sample_number]
					j = j+1
		encoded_samples = {}
		encoded_samples['Interaction'] = ZZ
		encoded_samples['Local'] = Z
	else: raise IOError('\'samples\' must be a numpy array')
	return encoded_samples

def TwoQubitGate(prog, two_q_arg, qubit_1, qubit_2):
	return prog.inst(CPHASE(4*two_q_arg, qubit_1, qubit_2)).inst(PHASE(-2*two_q_arg, qubit_1)).inst(PHASE(-2*two_q_arg, qubit_2))

def IQPLayer(prog, qubits, phi_Z, phi_ZZ):

	for j in qubits:
		#Apply local Z rotations (b) to each qubit
		#If the particular qubit sample == 0, apply no gate
		if (phi_Z[j] != False):
			prog.inst(PHASE(-2*phi_Z[j],j))
		#Apply Control-Phase(Phi_ZZ_1) gates to each qubit
		for i in qubits: 
			if (qubits[i] < qubits[j]):
				#If the particular qubit sample pair == 0, apply no gate
				if (phi_ZZ[i,j] != False):
					prog = TwoQubitGate(prog, phi_ZZ[i,j], i,j)
	return prog
	
def KernelCircuit(device_params, phi_ZZ_1, phi_Z_1, phi_ZZ_2, phi_Z_2):
	'''Compute Quantum kernel given samples from the Born Machine (born_samples) and the Data Distribution (data_samples)
		This must be done for every sample from each distribution (batch gradient descent), (x, y)'''
	'''First layer, sample from first distribution (1), parameters phi_ZZ_1, phi_Z_1'''
	device_name = device_params[0]
	as_qvm_value = device_params[1]

	qc = get_qc(device_name, as_qvm = as_qvm_value)
	qubits = qc.qubits()
	prog = Program()

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

def QuantumKernelComputation(device_params, N_samples1, N_samples2, N_kernel_samples, encoded_samples1, encoded_samples2):
	kernel_approx = np.zeros((N_samples1, N_samples2))
	kernel_exact = np.zeros((N_samples1, N_samples2))
	#define a dictionary for both approximate and exact kernel
	kernel_dict = {}
	kernel_exact_dict = {}
	device_name = device_params[0]
	as_qvm_value = device_params[1]

	qc = get_qc(device_name, as_qvm = as_qvm_value)
	qubits = qc.qubits()
	N_qubits = len(qubits)
	make_wf = WavefunctionSimulator()
	for sample2 in range(0, N_samples2):
		for sample1 in range(0, sample2+1):

			s_temp1 = ConvertToString(sample1, N_qubits)
			s_temp2 = ConvertToString(sample2, N_qubits)
			ZZ_1 = encoded_samples1['Interation']
			ZZ_2 = encoded_samples2['Interation']

			Z_1 = encoded_samples1['Local']
			Z_2 = encoded_samples2['Local']

			prog = KernelCircuit(qubits, ZZ_1[:,:,sample1], Z_1[:,sample1], ZZ_2[:,:,sample2], Z_2[:,sample2])
			kernel_outcomes = make_wf.wavefunction(prog).get_outcome_probs()

			#Create zero string
			zero_string = '0'*N_qubits
			kernel_exact[sample1, sample2] = kernel_outcomes[zero_string]
			kernel_exact[sample2, sample1] = kernel_exact[sample2, sample1]
			kernel_exact_dict[s_temp1, s_temp2] = kernel_exact[sample1, sample2]
			kernel_exact_dict[s_temp2, s_temp1] = kernel_exact_dict[s_temp1, s_temp2]

		
			if (N_kernel_samples == 'infinite'):
				#If the kernel is computed exactly, just fill a dictionary with the exact values for each sample
				kernel_approx[sample1,sample2] = kernel_exact[sample1, sample2]
				kernel_approx[sample2,sample1] = kernel_approx[sample1,sample2]
				kernel_dict[s_temp1, s_temp2] = kernel_exact_dict[s_temp1, s_temp2]
				kernel_dict[s_temp2, s_temp1] = kernel_dict[s_temp1, s_temp2]

			else:

				#Index list for classical registers we want to put measurement outcomes into.
				#Measure the kernel circuit to compute the kernel approximately, the kernel is the probability of getting (00...000) outcome.
				#All (5) qubits are measured at once into dictionary, convert into array
				kernel_measurements_all_qubits_dict = qc.run_and_measure(prog, N_kernel_samples)
				kernel_measurements_used_qubits = np.flip(np.vstack(kernel_measurements_all_qubits_dict[q] for q in sorted(qubits)).T, 1)

				#qc.run_and_measure measures ALL (5) qubits, remove measurements which are not needed on qubits (N_qubits, 5]
				# kernel_measurements_used_qubits = np.flip(np.delete(kernel_measurements_all_qubits, range(N_qubits, 5), axis=1), 1)
				#m is total number of samples, n is the number of used qubits (out of 9)
				(m,n) = kernel_measurements_used_qubits.shape

				N_zero_strings = m - np.count_nonzero(np.count_nonzero(kernel_measurements_used_qubits, 1))
				#The kernel is given by = [Number of times outcome (00...000) occurred]/[Total number of measurement runs]

				kernel_approx[sample1,sample2] = N_zero_strings/N_kernel_samples
				#kernel is symmetric in (x,y) -> k(x,y)
				kernel_approx[sample2,sample1] = kernel_approx[sample1,sample2]
				kernel_dict[s_temp1, s_temp2] = kernel_approx[sample1, sample2]
				kernel_dict[s_temp2, s_temp1] = kernel_dict[s_temp1, s_temp2]

	return kernel_approx, kernel_exact, kernel_dict, kernel_exact_dict
