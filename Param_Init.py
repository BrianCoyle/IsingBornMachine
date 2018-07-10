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


'''This function computes the initial parameter values, J, b randomly chosen on interval [0, pi/4], gamma set to constant = pi/4'''
'''It also computes the state produced after the QAOA circuit. ''' 

#Initialise weights and biases as random
def network_params(N, J, b, gamma_x, gamma_y):
	for j in range(0,N):	
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
			while (i<j):
				J[i][j] = uniform(0,pi/4)
				J[j][i] = J[i][j]
				i = i+1

	return J, b, gamma_x, gamma_y


#Initialise Quantum State created after application of gate sequence
def state_init(N, N_v, N_h, J, b,  gamma_x, gamma_y, p, q, r, final_layer, control, sign):
		
		#sign = 0 for the positive probability version, sign =1 for the negative version of the probability (only used to compute the gradients)
		#final_layer is either 0, 1 or 2, for IQP (Final Hadamard), QAOA (Final X rotation) or [INSERT COOL NAME] (Final Y rotation)
		#control = 0 for updating biases, = 1 for updating weights, =2 for neither
		prog = Program()
		qvm = QVMConnection()

		for qubit_index in range(0,N):
			prog.inst(H(qubit_index)) 

		#Apply Control-Phase(J) gates to each qubit
		for j in range(0,N):
			i = 0
			while (i < j):
				if (control == 1 and i == p and j == q and sign == 0):
					prog.inst(CPHASE(J[i, j] + pi/2,i,j))
				elif (control == 1 and i == p and j == q and sign == 1):
					prog.inst(CPHASE(J[i, j] - pi/2,i,j))
				elif (control==2 and sign == 2):
					prog.inst(CPHASE(J[i, j],i,j))
				i = i+1
	
		#Apply local Z rotations (b) to each qubit (with one phase changed by pi/2 if the corresponding parameter {r} is being updated		
		for j in range(0,N):
			if (control == 0 and j == r and sign == 0):
				prog.inst(PHASE(b[j] + pi/2,j))
			elif (control == 0 and j == r and sign == 1):
				prog.inst(PHASE(b[j] - pi/2,j))
			elif (control==2 and sign == 2):
				prog.inst(PHASE(b[j],j))

		#Apply final layer to all qubits, either all Hadamard, or X or Y rotations

		if final_layer == 0:
			for k in range(0,N):
				prog.inst(H(k))

		elif final_layer == 1:
			#Apply e^(-i(pi/4)X_i) to each qubit
			for k in range(0,N):
				H_temp = (-float(gamma_x[k]))*sX(k)
				prog.inst(pl.exponential_map(H_temp)(1.0))

		elif final_layer == 2:
			#Apply e^(-i(pi/4)Y_i) to each qubit
			for k in range(0,N):
				H_temp = (-float(gamma_y[k]))*sY(k)
				prog.inst(pl.exponential_map(H_temp)(1.0))

		else: print("final_layer must be either 0, 1 or 2, for IQP (Final Hadamard), QAOA (Final X rotation) or [INSERT COOL NAME] (Final Y rotation)")

		wavefunction = qvm.wavefunction(prog)
		probs = qvm.wavefunction(prog).get_outcome_probs()
		
		return prog, wavefunction, probs

	
