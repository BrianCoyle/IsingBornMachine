from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
import numpy as np
from numpy import pi, log2
from pyquil.api import QVMConnection
from random import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin
import matplotlib.pyplot as plt

from Train_Generation import training_data, data_sampler
from Param_Init import network_params, state_init
from Sample_Gen import born_sampler



def kernel_circuit(phi_ZZ_1, phi_Z_1, phi_ZZ_2, phi_Z_2, N_v):
	'''Compute kernel given samples from the Born Machine (born_samples) and the Data Distribution (data_samples)
		This must be done for every sample from each distribution (batch gradient descent), (x, y)'''

	'''First layer, sample from first distribution (1), parameters phi_ZZ_1, phi_Z_1'''

	qvm = QVMConnection()
	prog = Program()

	for qubit_index in range(0, N_v):
		prog.inst(H(qubit_index))

	for j in range(0,N_v):
		#Apply local Z rotations (b) to each qubit
		if (int(phi_Z_1[j]) != 0):
			prog.inst(PHASE(phi_Z_1[j],j))

		#Apply Control-Phase(J) gates to each qubit
		i = 0
		while (i < j):
			if (int(phi_ZZ_1[i,j]) != 0):
				prog.inst(CPHASE(phi_ZZ_1[i,j],i,j))
			i = i+1

	'''Second layer, sample from both distributions with parameters, phi_ZZ_1 - phi_ZZ_2, phi_Z_1 - phi_Z_2'''

	for qubit_index in range(0, N_v):
			prog.inst(H(qubit_index))

	for j in range(0,N_v):

		#Apply local Z rotations (b) to each qubit
		if (int(phi_Z_1[j]-phi_Z_2[j]) != 0):
			prog.inst(PHASE(phi_Z_1[j]-phi_Z_2[j],j))

		#Apply Control-Phase(J) gates to each qubit
		i = 0
		while (i < j):
			if (int(phi_ZZ_1[i,j]-phi_ZZ_2[i,j]) != 0):
				prog.inst(CPHASE(phi_ZZ_1[i,j]-phi_ZZ_2[i,j],i,j))
			i = i+1


	'''Third Layer, sample from Data distibution, (y)'''
	for qubit_index in range(0, N_v):
			prog.inst(H(qubit_index))


	for j in range(0,N_v):
		#Apply local Z rotations (b) to each qubit
		if (int(phi_Z_2[j]) != 0):
			prog.inst(PHASE(-phi_Z_2[j],j))

		i = 0
		#Apply Control-Phase(J) gates to each qubit
		while (i < j):
			if (int(phi_ZZ_2[i, j]) != 0):
				prog.inst(CPHASE(-phi_ZZ_2[i,j],i,j))
			i = i+1

	for qubit_index in range(0, N_v):
			prog.inst(H(qubit_index))


	return prog
