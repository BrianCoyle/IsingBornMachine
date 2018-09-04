from pyquil.quil import Program
from pyquil.paulis import *
from pyquil.gates import *
import numpy as np
from pyquil.api import QVMConnection
from random import *
import matplotlib.pyplot as plt

from train_generation import TrainingData, DataSampler, ConvertToString
from param_init import NetworkParams, StateInit, HadamardToAll
from sample_gen import BornSampler

'''Define Non-Linear function to encode samples in graph weights/biases'''
#Fill graph weight and bias arrays according to one single sample
def EncodingFunc(N_v, samples):
	N_samples = samples.shape[0]
	ZZ = np.zeros((N_v, N_v, N_samples))
	Z = np.zeros((N_v, N_samples))
	for sample_number in range(0, N_samples):
		for i in range(0, N_v):
			if int(samples[sample_number, i]) == 1:
				Z[i, sample_number] = (np.pi)/4
			j = 0
			while (j < i):
				if int(samples[sample_number, i]) == 1 and int(samples[sample_number, j]) == 1:
					ZZ[i,j, sample_number] = (np.pi)/4
					ZZ[j,i, sample_number] = ZZ[i,j, sample_number]
				j = j+1
	return ZZ, Z

def TwoQubitGate(prog, two_q_arg, qubit_1, qubit_2):
	return prog.inst(CPHASE(4*two_q_arg,qubit_1, qubit_2)).inst(PHASE(-2*two_q_arg, qubit_1)).inst(PHASE(-2*two_q_arg, qubit_2))

def KernelCircuit(phi_ZZ_1, phi_Z_1, phi_ZZ_2, phi_Z_2, N_v):
	'''Compute Quantum kernel given samples from the Born Machine (born_samples) and the Data Distribution (data_samples)
		This must be done for every sample from each distribution (batch gradient descent), (x, y)'''
	'''First layer, sample from first distribution (1), parameters phi_ZZ_1, phi_Z_1'''
	qvm = QVMConnection()
	prog = Program()

	prog = HadamardToAll(prog, N_v)

	for j in range(0, N_v):
		one_q_arg = phi_Z_1[j]
	#Apply local Z rotations (b) to each qubit
		if (one_q_arg != False):
			prog.inst(PHASE(-2*one_q_arg,j))

	#Apply Control-Phase(Phi_ZZ_1) gates to each qubit
		i = 0
		while (i < j):
			two_q_arg = phi_ZZ_1[i,j]
			if (two_q_arg != False):
				prog = TwoQubitGate(prog, two_q_arg, i,j)
			i = i+1

	'''Second layer, sample from both distributions with parameters,
	 	phi_ZZ_1 - phi_ZZ_2, phi_Z_1 - phi_Z_2'''
	prog = HadamardToAll(prog, N_v)

	for j in range(0, N_v):
		one_q_arg = phi_Z_1[j]-phi_Z_2[j]
		#Apply local Z rotations (b) to each qubit
		if (one_q_arg  != False):
			prog.inst(PHASE(-one_q_arg ,j))
		#Apply Control-Phase(J) gates to each qubit
		i = 0
		while (i < j):
			two_q_arg = phi_ZZ_1[i,j]-phi_ZZ_2[i,j]
			if (two_q_arg !=  False):
				prog = TwoQubitGate(prog, -two_q_arg, i,j)
			i = i+1

	'''Third Layer, sample from Data distibution, (y)'''
	prog = HadamardToAll(prog, N_v)

	for j in range(0,N_v):
		one_q_arg = phi_Z_2[j]
		#Apply local Z rotations (b) to each qubit
		if (one_q_arg != False):
			prog.inst(PHASE(2*one_q_arg,j))
		i = 0
		#Apply Control-Phase(J) gates to each qubit
		while (i < j):
			two_q_arg = phi_ZZ_2[i, j]
			if (two_q_arg != False):
				prog = TwoQubitGate(prog, -two_q_arg, i,j)
			i = i+1

	prog = HadamardToAll(prog, N_v)
	return prog

def KernelComputation(N_v, N_samples1, N_samples2, N_kernel_samples, ZZ_1, Z_1, ZZ_2, Z_2):
	kernel = np.zeros((N_samples1, N_samples2))
	kernel_exact = np.zeros((N_samples1, N_samples2))
	#define a dictionary for both approximate and exact kernel
	kernel_dict = {}
	kernel_exact_dict = {}

	for sample2 in range(0, N_samples2):
		for sample1 in range(0, sample2+1):

			s_temp1 = ConvertToString(sample1, N_v)
			s_temp2 = ConvertToString(sample2, N_v)
			qvm = QVMConnection()
			prog = KernelCircuit(ZZ_1[:,:,sample1], Z_1[:,sample1], ZZ_2[:,:,sample2], Z_2[:,sample2], N_v)
			kernel_outcomes = qvm.wavefunction(prog).get_outcome_probs()

			#Create zero string
			zero_string = '0'*N_v
			kernel_exact[sample1, sample2] = kernel_outcomes[zero_string]
			kernel_exact[sample2, sample1] = kernel_exact[sample2, sample1]
			kernel_exact_dict[s_temp1, s_temp2] = kernel_exact[sample1, sample2]
			kernel_exact_dict[s_temp2, s_temp1] = kernel_exact_dict[s_temp1, s_temp2]

			#Index list for classical registers we want to put measurement outcomes into.
			#Measure the kernel circuit to compute the kernel approximately, the kernel is the probability of getting (00...000) outcome.
			if (N_kernel_samples == 'infinite'):
						kernel[sample1,sample2] = kernel_exact[sample1, sample2]
						kernel[sample2,sample1] = kernel[sample1,sample2]
						kernel_dict[s_temp1, s_temp2] = kernel_exact_dict[s_temp1, s_temp2]
						kernel_dict[s_temp2, s_temp1] = kernel_dict[s_temp1, s_temp2]
			else:
				classical_regs = list(range(0, N_v))

				for qubit_index in range(0, N_v):
					prog.measure(qubit_index, qubit_index)

				kernel_measurements = np.asarray(qvm.run(prog, classical_regs, N_kernel_samples))
				(m,n) = kernel_measurements.shape

				N_zero_strings = m - np.count_nonzero(np.count_nonzero(kernel_measurements, 1))
				#The kernel is given by = [Number of times outcome (00...000) occurred]/[Total number of measurement runs]

				kernel[sample1,sample2] = N_zero_strings/N_kernel_samples
				kernel[sample2,sample1] = kernel[sample1,sample2]
				kernel_dict[s_temp1, s_temp2] = kernel[sample1, sample2]
				kernel_dict[s_temp2, s_temp1] = kernel_dict[s_temp1, s_temp2]

	return kernel, kernel_exact, kernel_dict, kernel_exact_dict
