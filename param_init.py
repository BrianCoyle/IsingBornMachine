from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin
from pyquil.api import QVMConnection

import numpy as np
from random import *
from numpy import pi,log2

def HadamardToAll(prog, N_qubits):
	for qubit_index in range(0, N_qubits):
		prog.inst(H(qubit_index))
	return prog

'''This function computes the initial parameter values, J, b randomly chosen on interval [0, pi/4], gamma set to constant = pi/4'''
'''It also computes the state produced after the QAOA circuit. '''

#Initialise weights and biases as random
def NetworkParams(N, J, b, gamma_x, gamma_y):
	for j in range(0, N):
			b[j] = uniform(0,pi/4)

			#If gamma_y to be trained also and variable for each qubit
			#gamma_x[j] = uniform(0,pi/4)
			#If gamma_y to be trained also and variable for each qubit
			#gamma_y[j] = uniform(0,pi/4)

			#If gamma_x constant for all qubits
			gamma_x[j] = pi/4
			#If gamma_y constant for all qubits
			gamma_y[j] = pi/4

			i = 0
			while (i < j):
				J[i][j] = uniform(0, pi/4)
				J[j][i] = J[i][j]
				i = i+1

	return J, b, gamma_x, gamma_y


#Initialise Quantum State created after application of gate sequence
def StateInit(N, N_v, N_h, J, b,  gamma_x, gamma_y, p, q, r, final_layer, control, sign):

		#sign = 'POSITIVE' for the positive probability version, sign = 'NEGATIVE' for the negative version of the probability (only used to compute the gradients)
		#final_layer is either 'IQP', 'QAOA', 'IQPy' for IQP (Final Hadamard), QAOA (Final X rotation) or IQPy (Final Y rotation)
		#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, = 'NEITHER' for neither
		prog = Program()
		qvm = QVMConnection()

		prog = HadamardToAll(prog, N)

		#Apply Control-Phase(4J) gates to each qubit, the factor of 4 comes from the decomposition of the Ising gate
		for j in range(0, N):
			i = 0
			while (i < j):
				if (control == 'WEIGHTS' and i == p and j == q and sign == 'POSITIVE'):
					prog.inst(CPHASE(4*J[i, j] + pi/2,i,j))
				elif (control == 'WEIGHTS' and i == p and j == q and sign == 'NEGATIVE'):
					prog.inst(CPHASE(4*J[i, j] - pi/2,i,j))
				elif (control== 'NEITHER' and sign == 'NEITHER'):
					prog.inst(CPHASE(4*J[i, j],i,j))
					prog.inst(PHASE(-2*J[i,j], i))
					prog.inst(PHASE(-2*J[i,j], j))
				i = i+1

		#Apply local Z rotations (b) to each qubit (with one phase changed by pi/2 if the corresponding parameter {r} is being updated
		for j in range(0,N):
			if (control == 'BIAS' and j == r and sign == 'POSITIVE'):
				prog.inst(PHASE(-2*b[j] + pi/2,j))
			elif (control == 'BIAS' and j == r and sign == 'NEGATIVE'):
				prog.inst(PHASE(-2*b[j] - pi/2,j))
			elif (control== 'NEITHER' and sign == 'NEITHER'):
				prog.inst(PHASE(-2*b[j],j))

		#Apply final 'measurement' layer to all qubits, either all Hadamard, or X or Y rotations

		if (final_layer == 'IQP'):
			#If the final 'measurement' layer is to be an IQP measurement (i.e. Hadamard on all qubits)
			prog = HadamardToAll(prog, N_qubits)
		elif (final_layer =='QAOA'):
			#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)X_i)on all qubits)
			for k in range(0,N):
				H_temp = (-float(gamma_x[k]))*sX(k)
				prog.inst(pl.exponential_map(H_temp)(1.0))
		elif (final_layer == 'IQPy' ):
			#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)Y_i)on all qubits)
			for k in range(0,N):
				H_temp = (-float(gamma_y[k]))*sY(k)
				prog.inst(pl.exponential_map(H_temp)(1.0))

		else: print("final_layer must be either 'IQP', 'QAOA' OR 'IQPy', for IQP (Final Hadamard), \
					QAOA (Final X rotation) or IQPy (Final Y rotation)")

		wavefunction = qvm.wavefunction(prog)
		outcome_dict = qvm.wavefunction(prog).get_outcome_probs()

		return prog, wavefunction, outcome_dict
