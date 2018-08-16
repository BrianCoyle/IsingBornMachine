from pyquil.quil import Program
from pyquil.paulis import *
from pyquil.gates import *
import numpy as np
from pyquil.api import QVMConnection
from random import *
import matplotlib.pyplot as plt

from train_generation import TrainingData, DataSampler
from param_init import NetworkParams, StateInit, HadamardToAll
from sample_gen import BornSampler

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
