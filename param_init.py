## @package param_init
#
# Initialise some inputted variables

from pyquil.quil import Program
import pyquil.paulis as pl
from pyquil.gates import H, CPHASE, PHASE
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
# This function computes the initial parameter values, J, b randomly chosen on interval [0, pi/4], gamma, delta set to constant = pi/4 if untrained
#
# @param[in] qc The Rigetti QuantumComputer Object that is chosen, e.g. 'Aspen-1-2Q-B'
#
# @return initialised parameters
def NetworkParams(qc, random_seed):

	N_qubits = len(qc.qubits())

    #Initialise arrays for parameters

	J 							= np.zeros((N_qubits, N_qubits))
	[b, gamma, delta, sigma]	= [np.zeros((N_qubits)) for _ in range(4)]


    #Set random seed to be fixed for reproducibility, set random_seed differently depending on whether quantum data
	#is generated, or whether the actual Born machine is being used.
	rand.seed(random_seed)
	for j in range(0, N_qubits):
		b[j] = rand.uniform(0, pi/4)
		# If delta to be trained also and variable for each qubit
		# rand.seed(j+N)
		# gamma[j] = rand.uniform(0,pi/4)
		#If delta to be trained also and variable for each qubit
		#delta[j] = uniform(0,pi/4)

		gamma[j] = pi/4			#If gamma constant for all qubits
		delta[j] = pi/4  		#If delta constant for all qubits
		sigma[j] = pi/4 		#If sigma constant for all qubits

		for i in range(0, N_qubits):	
			if i < j:
				J[i][j] = rand.uniform(0, pi/4)
		J = J + np.transpose(J)

	initial_params = {'J': J, 'b': b, 'gamma': gamma, 'delta': delta, 'sigma': sigma}

	return initial_params

def NetworkParamsSingleQubitGates(qc, layers):
	'''This function initilises single-qubit trainable parameters'''

	N_qubits = len(qc.qubits())
	
	#Initialise arrays for parameters
	single_qubit_params	= np.zeros(N_qubits, N_qubits, N_qubits, layers) 
	#layers is the number of single qubit layers, each 'layer', l, consists of three gates, R_z(\theta_l^1)R_x(\theta_l^2)R_x(\theta_l^3)

	#Set random seed to be fixed for reproducibility
	rand.seed(0)
	for j in range(0, N_qubits):
		for l in range(0, layers):
			#initialise all single qubit gates at random
			single_qubit_params[j, :, :, l] = rand.uniform(0,pi/4)
			single_qubit_params[:, j, :, l] = rand.uniform(0,pi/4)
			single_qubit_params[:, :, j, l] = rand.uniform(0,pi/4)

	return single_qubit_params

# Initialise Quantum State created after application of gate sequence
def StateInit(qc, circuit_params, p, q, r, s, circuit_choice, control, sign):
		'''This function computes the state produced after the given circuit, either QAOA, IQP, or IQPy,
		depending on the value of circuit_choice.'''

		#sign = 'POSITIVE' for the positive probability version, sign = 'NEGATIVE' for the negative version of the probability (only used to compute the gradients)
		#final_layer is either 'IQP', 'QAOA', 'IQPy' for IQP (Final Hadamard), QAOA (Final X rotation) or IQPy (Final Y rotation)
		#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, = 'NEITHER' for neither

		#Initialise empty quantum program, with QuantumComputer Object, and Wavefunction Simulator
		prog = Program()
		
		qubits = qc.qubits()
		N_qubits = len(qubits)
		#Unpack circuit parameters from dictionary
		J = circuit_params['J']
		b = circuit_params['b']
		gamma = circuit_params['gamma']
		delta = circuit_params['delta']

		#Apply hadarmard to all qubits in computation
		prog = HadamardToAll(prog, qubits)

		#Apply Control-Phase(4J) gates to each qubit, the factor of 4 comes from the decomposition of the Ising gate
		#with local Z corrections to neighbouring qubits, coming from the decomposition of the Ising gate
		#If weight J_{p,q} is updated, add a +/- pi/2 rotation
		for j in range(0, N_qubits):
			for i in range(0, N_qubits):
					if (i < j): #connection is symmetric, so don't overcount entangling gates
						if (control == 'WEIGHTS' and i == p and j == q):
							prog.inst(CPHASE(4*J[i, j] + (-1)**(sign)*pi/2, qubits[i], qubits[j]))
							prog.inst(PHASE(-2*J[i, j] + (-1)**(sign)*pi/2, qubits[i]))
							prog.inst(PHASE(-2*J[i, j] + (-1)**(sign)*pi/2, qubits[j]))
		
						elif (control== 'NEITHER' or 'BIAS' or 'GAMMA' and sign == 'NEITHER'):
							prog.inst(CPHASE(4*J[i, j], qubits[i], qubits[j]))
							prog.inst(PHASE(-2*J[i, j], qubits[i]))
							prog.inst(PHASE(-2*J[i, j], qubits[j]))		

		#Apply local Z rotations (b) to each qubit (with one phase changed by pi/2 if the corresponding parameter {r} is being updated
		for j in range(0, N_qubits):
			if (control == 'BIAS' and j == r):
				prog.inst(PHASE(-2*b[j] +(-1)**(sign)*pi/2, qubits[j]))
			elif (control== 'NEITHER' or 'WEIGHTS' or 'GAMMA' and sign == 'NEITHER'):
				prog.inst(PHASE(-2*b[j], 		qubits[j]))
				
		#Apply final 'measurement' layer to all qubits, either all Hadamard, or X or Y rotations
		if (circuit_choice == 'IQP'):
			prog = HadamardToAll(prog, qubits) 	#If the final 'measurement' layer is to be an IQP measurement (i.e. Hadamard on all qubits)

		elif (circuit_choice =='QAOA'):
			#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)X_i)on all qubits)
			for k in range(0, N_qubits):
				# if (control == 'GAMMA' and k == s):
				# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma[k])+ (-1)**(sign)*pi/2))
	
				# elif (control == 'NEITHER' or 'WEIGHTS' or 'BIAS' and sign == 'NEITHER'):
				H_temp = (-float(gamma[k]))*pl.sX(qubits[k])
				prog.inst(pl.exponential_map(H_temp)(1.0))
				# print('GAMMA IS:',-float(gamma[k]))
		elif (circuit_choice == 'IQPy' ):
			#If the final 'measurement' layer is to be a IQPy measurement (i.e. e^(-i(pi/4)Y_i) on all qubits)
			for k in qubits:
				H_temp = (-float(delta[k]))*pl.sY(qubits[k])
				prog.inst(pl.exponential_map(H_temp)(1.0))

		else: raise IOError("circuit_choice must be either  \
							\'IQP\' for IQP (Final Hadamard), \
							\'QAOA\' for QAOA (Final X rotation) or \
							\'IQPy\' IQPy (Final Y rotation)")
	
		# print(prog)
		return prog


# # Initialise Quantum State created after application of gate sequence
# def StateInit(qc, circuit_params, p, q, r, s, circuit_choice, control, sign):
# 		'''This function computes the state produced after the given circuit, either QAOA, IQP, or IQPy,
# 		depending on the value of circuit_choice.'''

# 		#sign = 'POSITIVE' for the positive probability version, sign = 'NEGATIVE' for the negative version of the probability (only used to compute the gradients)
# 		#final_layer is either 'IQP', 'QAOA', 'IQPy' for IQP (Final Hadamard), QAOA (Final X rotation) or IQPy (Final Y rotation)
# 		#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, = 'NEITHER' for neither

# 		#Initialise empty quantum program, with QuantumComputer Object, and Wavefunction Simulator
# 		prog = Program()
		
# 		qubits = qc.qubits()
# 		N_qubits = len(qubits)
# 		#Unpack circuit parameters from dictionary
# 		J = circuit_params['J']
# 		b = circuit_params['b']
# 		gamma = circuit_params['gamma']
# 		delta = circuit_params['delta']

# 		#Apply hadarmard to all qubits in computation
# 		prog = HadamardToAll(prog, qubits)

# 		#Apply Control-Phase(4J) gates to each qubit, the factor of 4 comes from the decomposition of the Ising gate
# 		#with local Z corrections to neighbouring qubits, coming from the decomposition of the Ising gate
# 		#If weight J_{p,q} is updated, add a +/- pi/2 rotation
# 		for j in range(0, N_qubits):
# 			for i in range(0, N_qubits):
# 				if (i < j): #connection is symmetric, so don't overcount entangling gates
# 					if (control == 'WEIGHTS' and i == p and j == q and sign == 'POSITIVE'):
# 						prog.inst(CPHASE(4*J[i, j] + pi/2, qubits[i], qubits[j]))
# 						prog.inst(PHASE(-2*J[i, j] + pi/2, qubits[i]))
# 						prog.inst(PHASE(-2*J[i, j] + pi/2, qubits[j]))
# 					elif (control == 'WEIGHTS' and i == p and j == q and sign == 'NEGATIVE'):
# 						prog.inst(CPHASE(4*J[i, j] - pi/2, qubits[i], qubits[j]))
# 						prog.inst(PHASE(-2*J[i, j] - pi/2, qubits[i]))
# 						prog.inst(PHASE(-2*J[i, j] - pi/2, qubits[j]))
# 					elif (control== 'NEITHER' or 'BIAS' or 'GAMMA' and sign == 'NEITHER'):
# 						prog.inst(CPHASE(4*J[i, j], qubits[i], qubits[j]))
# 						prog.inst(PHASE(-2*J[i, j], qubits[i]))
# 						prog.inst(PHASE(-2*J[i, j], qubits[j]))					


# 		#Apply local Z rotations (b) to each qubit (with one phase changed by pi/2 if the corresponding parameter {r} is being updated
# 		for j in range(0, N_qubits):
# 			if (control == 'BIAS' and j == r and sign == 'POSITIVE'):
# 				prog.inst(PHASE(-2*b[j] + pi/2, qubits[j]))
# 			elif (control == 'BIAS' and j == r and sign == 'NEGATIVE'):
# 				prog.inst(PHASE(-2*b[j] - pi/2, qubits[j]))
# 			elif (control== 'NEITHER' or 'WEIGHTS' or 'GAMMA' and sign == 'NEITHER'):
# 				prog.inst(PHASE(-2*b[j], 		qubits[j]))
				
# 		#Apply final 'measurement' layer to all qubits, either all Hadamard, or X or Y rotations
# 		if (circuit_choice == 'IQP'):
# 			#If the final 'measurement' layer is to be an IQP measurement (i.e. Hadamard on all qubits)
# 			prog = HadamardToAll(prog, qubits)
# 		elif (circuit_choice =='QAOA'):
# 			#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)X_i)on all qubits)
# 			for k in range(0, N_qubits):
# 				# if (control == 'GAMMA' and k == s and sign == 'POSITIVE'):
# 				# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma[k])+ pi/2))
# 				# elif (control == 'GAMMA' and k == s and sign == 'NEGATIVE'):
# 				# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma[k])- pi/2))
# 				# elif (control == 'NEITHER' or 'WEIGHTS' or 'BIAS' and sign == 'NEITHER'):
# 				H_temp = (-float(gamma[k]))*pl.sX(qubits[k])
# 				prog.inst(pl.exponential_map(H_temp)(1.0))
# 				# print('GAMMA IS:',-float(gamma[k]))
# 		elif (circuit_choice == 'IQPy' ):
# 			#If the final 'measurement' layer is to be a IQPy measurement (i.e. e^(-i(pi/4)Y_i) on all qubits)
# 			for k in qubits:
# 				H_temp = (-float(delta[k]))*pl.sY(qubits[k])
# 				prog.inst(pl.exponential_map(H_temp)(1.0))

# 		else: raise IOError("circuit_choice must be either  \
# 							\'IQP\' for IQP (Final Hadamard), \
# 							\'QAOA\' for QAOA (Final X rotation) or \
# 							\'IQPy\' IQPy (Final Y rotation)")
	

# 		return prog


# class IsingBornMachine:

# 	def __init__(self, qc, circuit_params):
# 		self.qc = qc
# 		self.circuit_params = circuit_params

	
# 	# Initialise Quantum State created after application of gate sequence
# 	def StateInit(self, qc, circuit_params, circuit_choice):
# 			'''This function computes the state produced after the given circuit, either QAOA, IQP, or IQPy,
# 			depending on the value of circuit_choice.'''

# 			#sign = 'POSITIVE' for the positive probability version, sign = 'NEGATIVE' for the negative version of the probability (only used to compute the gradients)
# 			#final_layer is either 'IQP', 'QAOA', 'IQPy' for IQP (Final Hadamard), QAOA (Final X rotation) or IQPy (Final Y rotation)
# 			#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, = 'NEITHER' for neither

# 			#Initialise empty quantum program, with QuantumComputer Object, and Wavefunction Simulator
# 			prog = Program()
			
# 			qubits = qc.qubits()
# 			N_qubits = len(qubits)
# 			#Unpack circuit parameters from dictionary
# 			J = circuit_params['J']
# 			b = circuit_params['b']
# 			gamma = circuit_params['gamma']
# 			delta = circuit_params['delta']

# 			#Apply hadarmard to all qubits in computation
# 			prog = HadamardToAll(prog, qubits)

# 			#Apply Control-Phase(4J) gates to each qubit, the factor of 4 comes from the decomposition of the Ising gate
# 			#with local Z corrections to neighbouring qubits, coming from the decomposition of the Ising gate
# 			#If weight J_{p,q} is updated, add a +/- pi/2 rotation
# 			for j in range(0, N_qubits):
# 				for i in range(0, N_qubits):
# 						if (i < j): #connection is symmetric, so don't overcount entangling gates
# 							# if (control == 'WEIGHTS' and i == p and j == q):
# 							# 	prog.inst(CPHASE(4*J[i, j] + (-1)**(sign)*pi/2, qubits[i], qubits[j]))
# 							# 	prog.inst(PHASE(-2*J[i, j] + (-1)**(sign)*pi/2, qubits[i]))
# 							# 	prog.inst(PHASE(-2*J[i, j] + (-1)**(sign)*pi/2, qubits[j]))
			
# 							# elif (control== 'NEITHER' or 'BIAS' or 'GAMMA' and sign == 'NEITHER'):
# 								prog.inst(CPHASE(4*J[i, j], qubits[i], qubits[j]))
# 								prog.inst(PHASE(-2*J[i, j], qubits[i]))
# 								prog.inst(PHASE(-2*J[i, j], qubits[j]))		

# 			#Apply local Z rotations (b) to each qubit (with one phase changed by pi/2 if the corresponding parameter {r} is being updated
# 			for j in range(0, N_qubits):
# 				# if (control == 'BIAS' and j == r):
# 				# 	prog.inst(PHASE(-2*b[j] +(-1)**(sign)*pi/2, qubits[j]))
# 				# elif (control== 'NEITHER' or 'WEIGHTS' or 'GAMMA' and sign == 'NEITHER'):
# 				prog.inst(PHASE(-2*b[j], 		qubits[j]))
					
# 			#Apply final 'measurement' layer to all qubits, either all Hadamard, or X or Y rotations
# 			if (circuit_choice == 'IQP'):
# 				prog = HadamardToAll(prog, qubits) 	#If the final 'measurement' layer is to be an IQP measurement (i.e. Hadamard on all qubits)

# 			elif (circuit_choice =='QAOA'):
# 				#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)X_i)on all qubits)
# 				for k in range(0, N_qubits):
# 					# if (control == 'GAMMA' and k == s):
# 					# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma[k])+ (-1)**(sign)*pi/2))
		
# 					# elif (control == 'NEITHER' or 'WEIGHTS' or 'BIAS' and sign == 'NEITHER'):
# 					H_temp = (-float(gamma[k]))*pl.sX(qubits[k])
# 					prog.inst(pl.exponential_map(H_temp)(1.0))
# 					# print('GAMMA IS:',-float(gamma[k]))
# 			elif (circuit_choice == 'IQPy' ):
# 				#If the final 'measurement' layer is to be a IQPy measurement (i.e. e^(-i(pi/4)Y_i) on all qubits)
# 				for k in qubits:
# 					H_temp = (-float(delta[k]))*pl.sY(qubits[k])
# 					prog.inst(pl.exponential_map(H_temp)(1.0))

# 			else: raise IOError("circuit_choice must be either  \
# 								\'IQP\' for IQP (Final Hadamard), \
# 								\'QAOA\' for QAOA (Final X rotation) or \
# 								\'IQPy\' IQPy (Final Y rotation)")
		

# 			return prog	

# 			# Initialise Quantum State created after application of gate sequence
# 	def GradStateInit(self, qc, circuit_params, p, q, r, s, circuit_choice, control, sign):
# 			'''This function computes the state produced after the given circuit, either QAOA, IQP, or IQPy,
# 			depending on the value of circuit_choice.'''

# 			#sign = 'POSITIVE' for the positive probability version, sign = 'NEGATIVE' for the negative version of the probability (only used to compute the gradients)
# 			#final_layer is either 'IQP', 'QAOA', 'IQPy' for IQP (Final Hadamard), QAOA (Final X rotation) or IQPy (Final Y rotation)
# 			#control = 'BIAS' for updating biases, = 'WEIGHTS' for updating weights, = 'NEITHER' for neither

# 			#Initialise empty quantum program, with QuantumComputer Object, and Wavefunction Simulator
# 			prog = Program()
			
# 			qubits = qc.qubits()
# 			N_qubits = len(qubits)
# 			#Unpack circuit parameters from dictionary
# 			J = circuit_params['J']
# 			b = circuit_params['b']
# 			gamma = circuit_params['gamma']
# 			delta = circuit_params['delta']

# 			#Apply hadarmard to all qubits in computation
# 			prog = HadamardToAll(prog, qubits)

# 			#Apply Control-Phase(4J) gates to each qubit, the factor of 4 comes from the decomposition of the Ising gate
# 			#with local Z corrections to neighbouring qubits, coming from the decomposition of the Ising gate
# 			#If weight J_{p,q} is updated, add a +/- pi/2 rotation
# 			for j in range(0, N_qubits):
# 				for i in range(0, N_qubits):
# 						if (i < j): #connection is symmetric, so don't overcount entangling gates
# 							if (control == 'WEIGHTS' and i == p and j == q):
# 								prog.inst(CPHASE(4*J[i, j] + (-1)**(sign)*pi/2, qubits[i], qubits[j]))
# 								prog.inst(PHASE(-2*J[i, j] + (-1)**(sign)*pi/2, qubits[i]))
# 								prog.inst(PHASE(-2*J[i, j] + (-1)**(sign)*pi/2, qubits[j]))
			
# 							elif (control== 'NEITHER' or 'BIAS' or 'GAMMA' and sign == 'NEITHER'):
# 								prog.inst(CPHASE(4*J[i, j], qubits[i], qubits[j]))
# 								prog.inst(PHASE(-2*J[i, j], qubits[i]))
# 								prog.inst(PHASE(-2*J[i, j], qubits[j]))		

# 			#Apply local Z rotations (b) to each qubit (with one phase changed by pi/2 if the corresponding parameter {r} is being updated
# 			for j in range(0, N_qubits):
# 				if (control == 'BIAS' and j == r):
# 					prog.inst(PHASE(-2*b[j] +(-1)**(sign)*pi/2, qubits[j]))
# 				elif (control== 'NEITHER' or 'WEIGHTS' or 'GAMMA' and sign == 'NEITHER'):
# 					prog.inst(PHASE(-2*b[j], 		qubits[j]))
					
# 			#Apply final 'measurement' layer to all qubits, either all Hadamard, or X or Y rotations
# 			if (circuit_choice == 'IQP'):
# 				prog = HadamardToAll(prog, qubits) 	#If the final 'measurement' layer is to be an IQP measurement (i.e. Hadamard on all qubits)

# 			elif (circuit_choice =='QAOA'):
# 				#If the final 'measurement' layer is to be a QAOA measurement (i.e. e^(-i(pi/4)X_i)on all qubits)
# 				for k in range(0, N_qubits):
# 					# if (control == 'GAMMA' and k == s):
# 					# 	prog.inst(pl.exponential_map(sX(k))(-float(gamma[k])+ (-1)**(sign)*pi/2))
		
# 					# elif (control == 'NEITHER' or 'WEIGHTS' or 'BIAS' and sign == 'NEITHER'):
# 					H_temp = (-float(gamma[k]))*pl.sX(qubits[k])
# 					prog.inst(pl.exponential_map(H_temp)(1.0))
# 					# print('GAMMA IS:',-float(gamma[k]))
# 			elif (circuit_choice == 'IQPy' ):
# 				#If the final 'measurement' layer is to be a IQPy measurement (i.e. e^(-i(pi/4)Y_i) on all qubits)
# 				for k in qubits:
# 					H_temp = (-float(delta[k]))*pl.sY(qubits[k])
# 					prog.inst(pl.exponential_map(H_temp)(1.0))

# 			else: raise IOError("circuit_choice must be either  \
# 								\'IQP\' for IQP (Final Hadamard), \
# 								\'QAOA\' for QAOA (Final X rotation) or \
# 								\'IQPy\' IQPy (Final Y rotation)")
		

# 			return prog	