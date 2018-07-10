from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
import numpy as np
from numpy import pi
from pyquil.api import QVMConnection
from random import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin

qvm = QVMConnection()
p = Program()
	
#This function computes the Amplitude of <0|U|0>

def Amplitude_Computation1(N, N_v, N_h, J, b, gamma, visibles, hiddens):
	''' Gate Definitions'''
	#Define Control-Hadamard
	a = 1/(np.sqrt(2))
	ch = np.array([[1, 0, 0, 0], 
					[0, 1, 0, 0], 
					[0, 0, a, a] ,
					[0, 0, a , -a]])

					
	# Define Control-Control-Phase with parameter thetaJ
	thetaJ = Parameter('thetaJ')
	ccphase = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, quil_exp(1j*thetaJ)]])
					
	#Define Control- e^(-i(gamma)X_i)
	thetagamma = Parameter('thetagamma')
	crx = np.array([[1, 0, 0, 0], 
					[0, 1, 0, 0], 
					[0, 0, quil_cos(thetagamma), 1j*quil_sin(thetagamma)] , 
					[0, 0, 1j*quil_sin(thetagamma) , quil_cos(thetagamma)]])
			
	#Define Control-X^z with parameter z, if z = 1, C-X applied, else Identity is applied.
	cx = np.array([[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 0, 1],
					[0, 0, 1, 0]])
					
	dg1 = DefGate('CH', ch)
	CH = dg1.get_constructor()

	dg2 = DefGate('CCPHASE', ccphase,[thetaJ])
	CCPHASE = dg2.get_constructor()

	dg3 = DefGate('CRX', crx, [thetagamma])
	CRX = dg3.get_constructor()

	dg4 = DefGate('CX', cx)
	CX = dg4.get_constructor()
	
	def Amplitude_Computation_Im(N, N_v, N_h, J, b,  gamma, prog, visibles, hiddens):
		''' Initial Ancilla Hadamard'''
		#Apply final Hadamard on auxiliary qubit
		prog.inst(H(N))
		prog.inst(PHASE(3*pi/2, N))

		
		'''Hadamard Gates'''
		#Apply Control-H over all qubits except first, controlled on the ancilla qubit
		for qtarget in range(0,N):
			prog.inst(CH(N, qtarget))
			
		'''Entanglement Gates'''
		#Apply Control-Control-Phase(J) to all pairs of qubits conditioned on ancilla
		for j in range(0,N):
			i = 0
			while (i < j):
				prog.inst(CCPHASE(J[i][j])( N, i, j))
				i = i+1
			
		'''Z - Rotations'''
		#Apply Control-Phase(b) to all qubits conditioned on ancilla
		for j in range(0,N):
			prog.inst(CPHASE(b[j],N, j))
			
		''' X Rotations'''
		#Apply Control-e^(-i(pi/4)X_i) to all qubits conditioned on ancilla
		for j in range(0,N):
			prog.inst(CRX(gamma[j])(N, j))
		
		'''Control - Measurement Outcomes'''
		#Apply Control-e^(-i(pi/4)X_i) to all qubits conditioned on ancilla
		for v in range(0,N_v):
			if (int(visibles[v]) == 1):
					prog.inst(CX(N, v))
		for h in range(0, N_h):	
			if (int(hiddens[h]) == 1):
					prog.inst(CX(N, h + N_v))

		''' Final Ancilla Hadamard'''
		#Apply Final Hadamard on Ancilla
		prog.inst(H(N))
	
		return prog
			
	def Amplitude_Computation_Real(N, N_v, N_h, J, b, gamma, prog, visibles, hiddens):

		''' Initial Ancilla Hadamard'''
		#Apply final Hadamard on auxiliary qubit
		prog.inst(H(N))
		
		'''Hadamard Preparation Gates'''
		#Apply Control-H over all qubits, controlled on the ancilla qubit
		for qtarget in range(0,N):
			prog.inst(CH(N, qtarget))

		'''Entanglement Gates'''
		#Apply Control-Control-Phase(J) to all pairs of qubits conditioned on ancilla
		for j in range(0,N):
			i = 0
			while (i < j):
				prog.inst(CCPHASE(J[i, j])(N, i, j))
				i = i+1
		
		'''Z Rotations'''
		#Apply Control-Phase(b) to all qubits conditioned on ancilla
		for j in range(0,N):
			prog.inst(CPHASE(b[j], N, j))

		
		''' X Rotations'''
		#Apply Control-e^(-i(pi/4)X_i) to all qubits conditioned on ancilla
		for j in range(0,N):
			prog.inst(CRX(gamma[j])(N, j))
			
	
		''' Measurement Outcomes'''
		#Apply Control-X conditioned on whether the measurement is 0/1 for qubit k
		for v in range(0,N_v):
			if (int(visibles[v]) == 1):
					prog.inst(CX(N, v))		
		for h in range(0, N_h):	
			if (int(hiddens[h]) == 1):
					prog.inst(CX(N, h + N_v))
		
		''' Final Ancilla Hadamard'''
		#Apply final Hadamard on auxiliary qubit
		prog.inst(H(N))
				
		return prog
		
	def Compute_Amp_Im(N, N_v, N_h, J, b, gamma, prog, visibles, hiddens):
	#This fuction computes the real part of the amplitude = 
	#   Re(Amp) = 1 - 2*Prob(Anc = |0>) 

		Q = Amplitude_Computation_Im(N, N_v, N_h, J, b, gamma, prog, visibles, hiddens)
		
		wavefunction = qvm.wavefunction(Q)
		
		probs = wavefunction.get_outcome_probs() # extracts the probabilities of outcomes
		probs_list = list(probs.keys())
		
		probs_list_zero_outcome = []
		
		for key in range(0,len(probs_list)):
			if probs_list[key].startswith('0'):
				probs_list_zero_outcome.append(probs.get(probs_list[key]))
		
		prob_zero = sum(probs_list_zero_outcome)
		
		# print('The Probability of measuring the ancilla with outcome 0 is: ', 
				 # prob_zero, '\n')
		
		amp_im =  2*prob_zero - 1
				
		return amp_im	
	
	def Compute_Amp_Real(N, N_v, N_h, J, b, gamma, prog, visibles, hiddens):
	#This fuction computes the real part of the amplitude = 
	#   Re(Amp) = 2*Prob(Anc = |0>) - 1

		P = Amplitude_Computation_Real(N, N_v, N_h, J, b, gamma, prog, visibles, hiddens)

	
		wavefunction = qvm.wavefunction(P)
		
		# extracts the probabilities of outcomes into a dictionary, .values() for the values
		# .keys() for the binary outcomes
		
		probs = wavefunction.get_outcome_probs()
		probs_list = list(probs.keys())

		probs_list_zero_outcome = []
		
		#Extract the probabilities of the ancilla being |0>, ancilla is in the first index
		for key in range(0,len(probs_list)):
			if probs_list[key].startswith('0'):
				probs_list_zero_outcome.append(probs.get(probs_list[key]))
					
		#sum of all probabilities with the ancilla being zero gives total probability of 
		#ancilla in |0>
		prob_zero = sum(probs_list_zero_outcome)
		
		# print('The Probability of measuring the ancilla with outcome 0 is: ',
				# prob_zero, '\n')
		
		amp_real = 2*prob_zero - 1
		
		# print('The real part of the amplitude is:', amp_real, '\n')
		
		return amp_real
		
	q_im = Program()

	q_im.inst(dg1)
	q_im.inst(dg2)
	q_im.inst(dg3)
	q_im.inst(dg4)

	
	Im = Compute_Amp_Im(N, N_v, N_h, J, b,  gamma, q_im, visibles, hiddens)

	#--------------------------------------------------------------------
	
	q_real = Program()

	q_real.inst(dg1)
	q_real.inst(dg2)
	q_real.inst(dg3)
	q_real.inst(dg4)

	
	Re = Compute_Amp_Real(N, N_v, N_h, J, b, gamma, q_real, visibles, hiddens)
	
	Amp1 = Re +1j*Im

	
	return Amp1


#This function computes the Amplitude of <0|U'|0>, where U'
#contains the gradient of the Hamiltonian

def Amplitude_Computation2(N, N_v, N_h, J, b, gamma, p, q, r,  visibles, hiddens, control):

	''' Gate Definitions'''
	#Define Control-Hadamard
	a = 1/(np.sqrt(2))
	ch = np.array([[1, 0, 0, 0], 
					[0, 1, 0, 0], 
					[0, 0, a, a] ,
					[0, 0, a , -a]])


		
	# Define Control-Control-Phase with parameter thetaJ
	thetaJ = Parameter('thetaJ')
	ccphase = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 1, 0],
					[0, 0, 0, 0, 0, 0, 0, quil_exp(1j*thetaJ)]])
					

	#Define Control- e^(-i(gamma)X_i)
	thetagamma = Parameter('thetagamma')
	crxdag = np.array([[1, 0, 0, 0], 
					[0, 1, 0, 0], 
					[0, 0, quil_cos(thetagamma), -1j*quil_sin(thetagamma)] , 
					[0, 0, -1j*quil_sin(thetagamma) , quil_cos(thetagamma)]])

					
	#Define Control-X^z with parameter z, if z = 1, C-X applied, else Identity is applied.
	cx = np.array([[1, 0, 0, 0],
					[0, 1, 0, 0],
					[0, 0, 0, 1],
					[0, 0, 1, 0]])
					
	
	dg1 = DefGate('CH', ch)
	CH = dg1.get_constructor()


	dg2 = DefGate('CCPHASE', ccphase,[thetaJ])
	CCPHASE = dg2.get_constructor()

	dg3 = DefGate('CRXDAG', crxdag, [thetagamma])
	CRXDAG = dg3.get_constructor()

	dg4 = DefGate('CX', cx)
	CX = dg4.get_constructor()
	
	def Amplitude_Computation_Im(N, N_v, N_h, J, b, gamma,  p , q ,r, prog, visibles, hiddens, control):
		
		'''Ancilla Initialisation in Hadamard'''
		#Apply Hadamard on Ancilla qubit
		prog.inst(H(N))
		prog.inst(PHASE(3*pi/2, N))

		'''Measurement Outcomes'''
		#Apply Control-X^{\dagger} conditioned on whether the measurement is 0/1 for all visible and hidden qubits
		for v in range(0,N_v):
			if (int(visibles[v]) == 1):
					prog.inst(CX(N, v))
		for h in range(0, N_h):	
			if (int(hiddens[h]) == 1):
					prog.inst(CX(N, h + N_v))
		
		'''X rotations'''
		#Apply Control-e^(-i(pi/4)X_i)^Dag to all qubits conditioned on ancilla
		for j in range(0,N):
			prog.inst(CRXDAG(gamma[j])(N, j))
			
		'''control is a parameter that is 1 if we are updating weights,
				0 if we are updating biases. p (and/or q) is the index of 
				the paramater we are updating
		'''
		if (control == 1):
			#updating paramter J[p][q] implies an additional Z on p and Z on q
			#The operation is controlled based on the ancilla because of the 
			#amplitude calculation
			prog.inst(CZ(N, p),CZ(N, q))
		if (control == 0):
			#updating paramter b[r] implies an additional Z on qubit r
			prog.inst(CZ(N, r))
			
		'''Z Rotations'''
		#Apply Control-Phase(b)^{dagger} to all qubits conditioned on ancilla
		for j in range(0,N):
			prog.inst(CPHASE(-b[j], j, N))
				
		
		'''Entanglement Gates'''
		#Apply Control-Control-Phase(J)^{\dagger} to all pairs of qubits conditioned on ancilla
		for j in range(0,N):
			i = 0
			while (i < j):
				prog.inst(CCPHASE(-J[i, j])(i, j, N))
				i = i+1
				
			
		'''Control Hadamard'''
		#Apply Control-H over all qubits except first, controlled on the ancilla qubit
		for i in range(0,N):
			prog.inst(CH(N, i))
			
		'''Final Ancilla Hadamard'''
		#Apply final Hadamard on auxiliary qubit
		prog.inst(H(N))
		
		return prog
			
		
	def Amplitude_Computation_Real(N, N_v, N_h, J, b, gamma, p, q,r, prog, visibles, hiddens, control):
		
		'''Ancilla Initialisation in Hadamard'''
		#Apply Hadamard on Ancilla qubit
		prog.inst(H(N))
		
		'''Measurement Outcomes'''
		#Apply Control-X^{\dagger} conditioned on whether the measurement is 0/1 for all visible and hidden qubits
		for v in range(0,N_v):
			if (int(visibles[v]) == 1):
					prog.inst(CX(N, v))
		for h in range(0, N_h):	
			if (int(hiddens[h]) == 1):
					prog.inst(CX(N, h + N_v))
		
		'''X rotations'''
		#Apply Control-e^(-i(pi/4)X_i)^Dag to all qubits conditioned on ancilla
		for j in range(0,N):
			prog.inst(CRXDAG(gamma[j])(N, j))
			
			
		'''control is a parameter that is 1 if we are updating weights,
				0 if we are updating biases. p (and/or q) is the index of 
				the paramater we are updating
		'''
		if (control == 1):
			#updating paramter J[p][q] implies an additional Z on p and Z on q
			#The operation is controlled based on the ancilla because of the 
			#amplitude calculation
			prog.inst(CZ(N, p),CZ(N, q))
		if (control == 0):
			#updating paramter b[r] implies an additional Z on qubit r
			prog.inst(CZ(N, r))
			
		'''Z Rotations'''
		#Apply Control-Phase(b)^{dagger} to all qubits conditioned on ancilla
		for j in range(0,N):
			prog.inst(CPHASE(-b[j], j, N))
				
		
		'''Entanglement Gates'''
		#Apply Control-Control-Phase(J)^{\dagger} to all pairs of qubits conditioned on ancilla
		for j in range(0,N):
			i = 0
			while (i < j):
				prog.inst(CCPHASE(-J[i, j])(i, j, N))
				i = i+1
				
		'''Control Hadamard'''
		#Apply Control-H over all qubits, controlled on the ancilla qubit
		for i in range(0,N):
			prog.inst(CH(N, i))
			
		'''Final Ancilla Hadamard'''
		#Apply final Hadamard on auxiliary qubit
		prog.inst(H(N))
		return prog
		
	def Compute_Amp_Im(N, N_v, N_h, J, b, gamma,  p, q, r, prog, visibles, hiddens, control):
	#This fuction computes the real part of the amplitude = 
	#   Re(Amp) = 2*Prob(Anc = |0>)  - 1

		Q = Amplitude_Computation_Im(N, N_v, N_h, J, b, gamma, p, q, r,  prog, visibles, hiddens, control)

		wavefunction = qvm.wavefunction(Q)
		probs = wavefunction.get_outcome_probs() # extracts the probabilities of outcomes

	
		probs_list = list(probs.keys())
		
		probs_list_zero_outcome = []
		
		for key in range(0,len(probs_list)):
			if probs_list[key].startswith('0'):
				probs_list_zero_outcome.append(probs.get(probs_list[key]))

		prob_zero = sum(probs_list_zero_outcome)
		
		# print('The Probability of measuring the ancilla with outcome 0 is: ', 
				# prob_zero, '\n')
		
		amp_im = 2*prob_zero-1 
		
		
		return amp_im	
	
	def Compute_Amp_Real(N, N_v, N_h, J, b, gamma, p, q, r, prog, visibles, hiddens, control):
	#This fuction computes the real part of the amplitude = 
	#   Re(Amp) = 2*Prob(Anc = |0>) - 1

		#Output the program after U'|0> applied
		Q = Amplitude_Computation_Real(N, N_v, N_h, J, b, gamma, p, q, r,   prog, visibles, hiddens, control)

		wavefunction = qvm.wavefunction(Q)

		probs = wavefunction.get_outcome_probs() # extracts the probabilities of outcomes
		
		probs_list = list(probs.keys())
		
		probs_list_zero_outcome = []
		
		for key in range(0,len(probs_list)):
			if probs_list[key].startswith('0'):
				probs_list_zero_outcome.append(probs.get(probs_list[key]))

		prob_zero = sum(probs_list_zero_outcome)
		
		'''The Real Part of the Amplitude'''
		amp_real = 2*prob_zero - 1
			
		return amp_real
		
	q_im = Program()

	q_im.inst(dg1)
	q_im.inst(dg2)
	q_im.inst(dg3)
	q_im.inst(dg4)

	Im = Compute_Amp_Im(N, N_v, N_h, J, b, gamma,  p, q, r, q_im, visibles, hiddens, control)

	#--------------------------------------------------------------------
	
	q_real = Program()

	q_real.inst(dg1)
	q_real.inst(dg2)
	q_real.inst(dg3)
	q_real.inst(dg4)

	Re = Compute_Amp_Real(N, N_v, N_h, J, b, gamma,  p, q, r, q_real, visibles, hiddens, control)
	
	Amp2 = Re +1j*Im

	return Amp2













































# def Amp1_DirectCompl(N, N_v, N_h, J, b, n, gamma):
	# prog = Program()
	# #Define Control-Hadamard

	# thetagamma = Parameter('thetagamma')
	# rx = np.array([[ quil_cos(thetagamma), 1j*quil_sin(thetagamma)] , 
					# [1j*quil_sin(thetagamma) , quil_cos(thetagamma)]])

	# dg4 = DefGate('RX', rx, [thetagamma])
	# RX = dg4.get_constructor()

	# prog.inst(dg4)
	
	# '''Initial Hadarmard Preparation'''
	# #Apply H over all qubits
	# for qtarget in range(0,N):
		# prog.inst(H(qtarget))
		
	# '''Entanglement Gates'''
	# for j in range(0,N):
		# i = 0
		# while (i < j):
			# #Apply Control-Phase(J) to all pairs of qubits 
			# prog.inst(CPHASE(J[i][j][n])( i, j))
			# i = i+1
		
	# '''Z Rotations'''
	# for j in range(0,N):
		# #Apply Phase(b) to all qubits conditioned on ancilla
		# prog.inst(PHASE(b[j][n])(j)) 
		
	# '''X Rotations'''		
	# for j in range(0,N):
		# #Apply Control-e^(-i(pi/4)X_i) to all qubits conditioned on ancilla
		# prog.inst(RX(gamma[j][n])(j))
	
	# return prog

# def Compute_Amp_RRREEAAL(N, N_v, N_h, J, b, n, gamma):
	
		# Q = Amp1_DirectCompl(N, N_v, N_h, J, b, n, gamma)

		# wavefunction = qvm.wavefunction(Q)

		# probs = wavefunction.get_outcome_probs() # extracts the probabilities of outcomes
		# print('\n probs are: ', probs)
		
		
		# return probs
		







#print('\n The First Amplitude is: \n', Amp1, '\n')
# print('\n The Second Amplitude is: \n', Amp2, '\n')


# print('\n The Amplitude is: \n', Probs_Amplitudes, '\n')
# print('\n P_dist is: \n', Probs_Dist, '\n')	

# print('\n P_v is: \n', pvsum, '\n')	
		
# deltaJ = (Probs_Amplitudes.sum(axis = 0))



# print(training_data)
# print(training_data.sum())


