## @package param_init
#
# Initialise some inputted variables

from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin
from pyquil.api import get_qc, WavefunctionSimulator

import numpy as np
import random as rand
from numpy import pi,log2

def HadamardToAll(prog, qubits):
	#qubits is an ordered list of the qubits available in the chip
	for qubit_index in qubits:
		prog.inst(H(qubit_index))
	return prog


## Initialise weights and biases as random
#
# This function computes the initial parameter values, J, b randomly chosen on interval [0, pi/4], gamma_x, gamma_y set to constant = pi/4 if untrained
#
# @param[in] N_qubits The number of qubits
#
# @return initialised parameters
def NetworkParams(device_params, random_seed):

    qc = get_qc(device_params[0], as_qvm = device_params[1])
    qubits = qc.qubits()

    #Initialise arrays for parameters
    #for examples qubits = [5,6,7] (using qubits labelled 5,6,7 on chip) int(qubits[-1])+1 = 7 + 1 = 8 elements
    # if (qubits[0] > 0):
    J 		= np.zeros((int(qubits[-1])+1, int(qubits[-1])+1))
    b 		= np.zeros((int(qubits[-1])+1))
    gamma_x = np.zeros((int(qubits[-1])+1))
    gamma_y = np.zeros((int(qubits[-1])+1))

    #Set random seed to be fixed for reproducibility, set random_seed differently depending on whether quantum data
	#is generated, or whether the actual Born machine is being used.
    rand.seed(random_seed)
    for j in qubits:
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
    
        for i in qubits:
            if i < j:
                J[i][j] = rand.uniform(0, pi/4)
                J[j][i] = J[i][j]
 
    initial_params = {}
    initial_params['J'] = J
    initial_params['b'] = b
    initial_params['gamma_x'] = gamma_x
    initial_params['gamma_y'] = gamma_y

    return initial_params

def NetworkParamsSingleQubitGates(device_params, layers):
	'''This function initilises single-qubit trainable parameters'''

	device_name 	= device_params[0]
	as_qvm_value 	= device_params[1]
	qc 				= get_qc(device_name, as_qvm = as_qvm_value)
	qubits 			= qc.qubits()
	
	#Initialise arrays for parameters
	#for examples qubits = [5,6,7] (using qubits labelled 5,6,7 on chip) int(qubits[-1])+1 = 7+1 = 8 elements
	single_qubit_params	= np.zeros((int(qubits[-1])+1),(int(qubits[-1])+1),(int(qubits[-1])+1), layers) 
	#layers is the number of single qubit layers, each 'layer', l, consists of three gates, R_z(\theta_l^1)R_x(\theta_l^2)R_x(\theta_l^3)

	#Set random seed to be fixed for reproducibility
	rand.seed(0)
	for j in qubits:
		for l in range(0, layers):
			#initialise all single qubit gates at random
			single_qubit_params[j, :, :, l] = rand.uniform(0,pi/4)
			single_qubit_params[:, j, :, l] = rand.uniform(0,pi/4)
			single_qubit_params[:, :, j, l] = rand.uniform(0,pi/4)

	return single_qubit_params

#Initialise Quantum State created after application of gate sequence
def StateInit(device_params, circuit_params, p, q, r, s, circuit_choice, control, sign):
		'''This function computes the state produced after the given circuit, either QAOA, IQP, or IQPy,
		depending on the value of circuit_choice.'''

		#sign = 'POSITIVE' for the positive probability version, sign = 'NEGATIVE' for the negative version of the probability (only used to compute the gradients)
		#final_layer is either 'IQP', 'QAOA', 'IQPy' for IQP (Final Hadamard), QAOA (Final X rotation) or IQPy (Final Y rotation)
		#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, = 'NEITHER' for neither

		#Initialise empty quantum program, with QuantumComputer Object, and Wavefunction Simulator
		prog = Program()
		
		device_name = device_params[0]
		as_qvm_value = device_params[1]

		qc = get_qc(device_name, as_qvm = as_qvm_value)

		qubits = qc.qubits() 	#qubits is an ordered list of the qubits available in the chip

		#Unpack circuit parameters from dictionary
		J = circuit_params['J']
		b = circuit_params['b']
		gamma_x = circuit_params['gamma_x']
		gamma_y = circuit_params['gamma_y']

		#Apply hadarmard to all qubits in computation
		prog = HadamardToAll(prog, qubits)

		#Apply Control-Phase(4J) gates to each qubit, the factor of 4 comes from the decomposition of the Ising gate
		#with local Z corrections to neighbouring qubits, coming from the decomposition of the Ising gate
		#If weight J_{p,q} is updated, add a +/- pi/2 rotation
		for j in qubits:
			for i in qubits:
				if (i < j): #connection is symmetric, so don't overcount entangling gates
					if (control == 'WEIGHTS' and i == p and j == q and sign == 'POSITIVE'):
						prog.inst(CPHASE(4*J[i, j] + pi/2, i, j))
						prog.inst(PHASE(-2*J[i, j] + pi/2, i))
						prog.inst(PHASE(-2*J[i, j] + pi/2, j))
					elif (control == 'WEIGHTS' and i == p and j == q and sign == 'NEGATIVE'):
						prog.inst(CPHASE(4*J[i, j] - pi/2, i, j))
						prog.inst(PHASE(-2*J[i, j] - pi/2, i))
						prog.inst(PHASE(-2*J[i, j] - pi/2, j))
					elif (control== 'NEITHER' or 'BIAS' or 'GAMMA' and sign == 'NEITHER'):
						prog.inst(CPHASE(4*J[i, j], i, j))
						prog.inst(PHASE(-2*J[i, j], i))
						prog.inst(PHASE(-2*J[i, j], j))					


		#Apply local Z rotations (b) to each qubit (with one phase changed by pi/2 if the corresponding parameter {r} is being updated
		for j in qubits:
			if (control == 'BIAS' and j == r and sign == 'POSITIVE'):
				prog.inst(PHASE(-2*b[j] + pi/2,j))
			elif (control == 'BIAS' and j == r and sign == 'NEGATIVE'):
				prog.inst(PHASE(-2*b[j] - pi/2,j))
			elif (control== 'NEITHER' or 'WEIGHTS' or 'GAMMA' and sign == 'NEITHER'):
				prog.inst(PHASE(-2*b[j],j))

		#Apply final 'measurement' layer to all qubits, either all Hadamard, or X or Y rotations
		if (circuit_choice == 'IQP'):
			#If the final 'measurement' layer is to be an IQP measurement (i.e. Hadamard on all qubits)
			prog = HadamardToAll(prog, qubits)
		elif (circuit_choice =='QAOA'):
			#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)X_i)on all qubits)
			for k in qubits:
				# if (control == 'GAMMA' and k == s and sign == 'POSITIVE'):
				# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma_x[k])+ pi/2))
				# elif (control == 'GAMMA' and k == s and sign == 'NEGATIVE'):
				# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma_x[k])- pi/2))
				# elif (control == 'NEITHER' or 'WEIGHTS' or 'BIAS' and sign == 'NEITHER'):
				H_temp = (-float(gamma_x[k]))*sX(k)
				prog.inst(pl.exponential_map(H_temp)(1.0))
				# print('GAMMA IS:',-float(gamma_x[k]))
		elif (circuit_choice == 'IQPy' ):
			#If the final 'measurement' layer is to be a IQPy measurement (i.e. e^(-i(pi/4)Y_i) on all qubits)
			for k in qubits:
				H_temp = (-float(gamma_y[k]))*sY(k)
				prog.inst(pl.exponential_map(H_temp)(1.0))

		else: raise IOError("circuit_choice must be either  \
							\'IQP\' for IQP (Final Hadamard), \
							\'QAOA\' for QAOA (Final X rotation) or \
							\'IQPy\' IQPy (Final Y rotation)")
	

		return prog
