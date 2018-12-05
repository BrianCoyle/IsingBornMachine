from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin
from pyquil.api import get_qc, WavefunctionSimulator
import pyquil
import numpy as np
import random as rand
from numpy import pi,log2
from param_init import HadamardToAll

import matplotlib.pyplot as plt
import networkx as nx
from auxiliary_functions import EmpiricalDist
#Initialise weights and biases as random
def NetworkParams(N, J, b, gamma_x, gamma_y):
	#Set random seed to be fixed for reproducibility

	rand.seed(0)
	for j in range(0, N):
			# rand.seed(j)
			b[j] = rand.uniform(0, pi/4)
			# If gamma_y to be trained also and variable for each qubit
			# rand.seed(j+N)
			# gamma_x[j] = rand.uniform(0,pi/4)
			#If gamma_y to be trained also and variable for each qubit
			#gamma_y[j] = uniform(0,pi/4)

			# #If gamma_x constant for all qubits
			gamma_x[j] = pi/4
			#If gamma_y constant for all qubits
			gamma_y[j] = pi/4

			i = 0
			while (i < j):
				J[i][j] = rand.uniform(0, pi/4)
				J[j][i] = J[i][j]
				i = i+1

	return J, b, gamma_x, gamma_y


#Initialise Quantum State created after application of gate sequence
def GoogleCircuit():
    #sign = 'POSITIVE' for the positive probability version, sign = 'NEGATIVE' for the negative version of the probability (only used to compute the gradients)
    #final_layer is either 'IQP', 'QAOA', 'IQPy' for IQP (Final Hadamard), QAOA (Final X rotation) or IQPy (Final Y rotation)
    #control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, = 'NEITHER' for neither
    prog = Program()
    qc = get_qc("5q-qvm")
    print(pyquil.list_quantum_computers(qpus = False))
    graph = qc.qubit_topology()
    nx.draw(graph)
    plt.show()
    # py.iplot(qc.qubit_topology(), filename='networkx')
    #Apply initial layer of Hadamards

    prog = HadamardToAll(prog, 3)
    make_wf = WavefunctionSimulator()
    # x = []
    # for i in range(10000000):
    #     x.append(rand.randrange(0,3,1))
    #     # print(x[i])
    # print(EmpiricalDist(x, 1))

		#Apply Control-Phase(4J) gates to each qubit, the factor of 4 comes from the decomposition of the Ising gate
		# for j in range(0, N):
		# 	i = 0
		# 	while (i < j):

		# 		if (control == 'WEIGHTS' and i == p and j == q and sign == 'POSITIVE'):
		# 			prog.inst(CPHASE(4*J[i, j] + pi/2,i,j))
		# 			prog.inst(PHASE(-2*J[i, j] + pi/2, i))
		# 			prog.inst(PHASE(-2*J[i, j] + pi/2, j))
		# 		elif (control == 'WEIGHTS' and i == p and j == q and sign == 'NEGATIVE'):
		# 			prog.inst(CPHASE(4*J[i, j] - pi/2,i,j))
		# 			prog.inst(PHASE(-2*J[i, j] - pi/2, i))
		# 			prog.inst(PHASE(-2*J[i, j] - pi/2, j))
		# 		elif (control== 'NEITHER' or 'BIAS' or 'GAMMA' and sign == 'NEITHER'):
		# 			prog.inst(CPHASE(4*J[i, j],i,j))
		# 			prog.inst(PHASE(-2*J[i,j], i))
		# 			prog.inst(PHASE(-2*J[i,j], j))					
		# 		i = i+1

		# #Apply local Z rotations (b) to each qubit (with one phase changed by pi/2 if the corresponding parameter {r} is being updated
		# for j in range(0, N):

		# 	if (control == 'BIAS' and j == r and sign == 'POSITIVE'):
		# 		prog.inst(PHASE(-2*b[j] + pi/2,j))
		# 	elif (control == 'BIAS' and j == r and sign == 'NEGATIVE'):
		# 		prog.inst(PHASE(-2*b[j] - pi/2,j))
		# 	elif (control== 'NEITHER' or 'WEIGHTS' or 'GAMMA' and sign == 'NEITHER'):
		# 		prog.inst(PHASE(-2*b[j],j))

		# 	# print('B IS:', b[j])
		# #Apply final 'measurement' layer to all qubits, either all Hadamard, or X or Y rotations
		# if (circuit_choice == 'IQP'):
		# 	#If the final 'measurement' layer is to be an IQP measurement (i.e. Hadamard on all qubits)
		# 	prog = HadamardToAll(prog, N)
		# elif (circuit_choice =='QAOA'):
		# 	#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)X_i)on all qubits)
		# 	for k in range(0, N):
		# 		# if (control == 'GAMMA' and k == s and sign == 'POSITIVE'):
		# 		# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma_x[k])+ pi/2))
		# 		# elif (control == 'GAMMA' and k == s and sign == 'NEGATIVE'):
		# 		# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma_x[k])- pi/2))
		# 		# elif (control == 'NEITHER' or 'WEIGHTS' or 'BIAS' and sign == 'NEITHER'):
		# 		H_temp = (-float(gamma_x[k]))*sX(k)
		# 		prog.inst(pl.exponential_map(H_temp)(1.0))
		# 		# print('GAMMA IS:',-float(gamma_x[k]))
		# elif (circuit_choice == 'IQPy' ):
		# 	#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)Y_i)on all qubits)
		# 	for k in range(0,N):
		# 		H_temp = (-float(gamma_y[k]))*sY(k)
		# 		prog.inst(pl.exponential_map(H_temp)(1.0))

		# else: raise IOError("circuit_choice must be either 'IQP', 'QAOA' OR 'IQPy', for IQP (Final Hadamard), \
		# 			QAOA (Final X rotation) or IQPy (Final Y rotation)")
		# # print('J = ', J)
		# # print('b = ', b)

		# wavefunction = make_wf.wavefunction(prog)
		# outcome_dict = make_wf.wavefunction(prog).get_outcome_probs()
		# # print(control, sign, outcome_dict, prog)

		# return prog, wavefunction, outcome_dict

GoogleCircuit()