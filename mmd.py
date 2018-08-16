from pyquil.quil import Program
from pyquil.paulis import *
from pyquil.gates import *
import numpy as np
from pyquil.api import QVMConnection
from random import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin
from param_init import StateInit, NetworkParams

from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler
from classical_kernel import GaussianKernel, GaussianKernelExact
from file_operations_out import KernelDictFromFile
from kernel_circuit import KernelCircuit
import ast
import sys
import json

def ConvertToString(index, N_v):
	return "0" * (N_v-len(format(index,'b'))) + format(index,'b')

'''Define Non-Linear function to encode samples in graph weights/biases'''
#Fill graph weight and bias arrays according to one single sample
def EncodingFunc(N_v, sample, N_samples):
	ZZ = np.zeros((N_v, N_v, N_samples))
	Z = np.zeros((N_v, N_samples))
	for sample_number in range(0, N_samples):
		for i in range(0, N_v):
			if int(sample[sample_number, i]) == 1:
				Z[i, sample_number] = (np.pi)/4
			j = 0
			while (j < i):
				if int(sample[sample_number, i]) == 1 and int(sample[sample_number, j]) == 1:
					ZZ[i,j, sample_number] = (np.pi)/4
					ZZ[j,i, sample_number] = ZZ[i,j, sample_number]
				j = j+1
	return ZZ, Z

def KernelComputation(N_v, N_samples1, N_samples2, N_kernel_samples, ZZ_1, Z_1, ZZ_2, Z_2):
	kernel = np.zeros((N_samples1, N_samples2))
	kernel_exact = np.zeros((N_samples1, N_samples2))
	#define a dictionary for both approximate and exact kernel
	kernel_dict = {}
	kernel_exact_dict = {}

	for sample2 in range(0, N_samples2):
		for sample1 in range(0, sample2+1):
			#convert integer in elements to binary string and store in list binary_string_list
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
##############################################################################################################
#If Everything is to be computed using samples
##############################################################################################################
def MMDKernelforGradSampler(N_v, data, born_samples,
						born_samples_plus, born_samples_minus,
						N_k_samples, k_choice):

	N_d 		= data.shape[0]
	N_b 		= born_samples.shape[0]
	N_b_plus 	= born_samples_plus.shape[0]
	N_b_minus	= born_samples_minus.shape[0]

	if (k_choice == 'Gaussian'):
		sigma = np.array([0.25, 10, 1000])
		k_bornplus_born = GaussianKernel(born_samples_plus, born_samples, sigma)
		k_bornminus_born = GaussianKernel(born_samples_minus, born_samples, sigma)
		k_bornplus_data = GaussianKernel(born_samples_plus, data, sigma)
		k_bornminus_data = GaussianKernel(born_samples_minus, data, sigma)
	elif (k_choice == 'Quantum'):
		ZZ_b, Z_b = EncodingFunc(N_v, born_samples, N_b)
		ZZ_b_plus, Z_b_plus = EncodingFunc(N_v, born_samples_plus, N_b_plus)
		ZZ_b_minus, Z_b_minus = EncodingFunc(N_v, born_samples_minus, N_b_minus)
		ZZ_d, Z_d = EncodingFunc(N_v, data, N_d)
		k_bornplus_born, k_exact_bornplus_born, k_dict_bornplus_born, k_exact_dict_bornplus_born \
								= KernelComputation(N_v, N_b_plus, N_b, N_k_samples, ZZ_b_plus, Z_b_plus, ZZ_b, Z_b)

		k_bornminus_born, k_exact_bornminus_born, k_dict_bornminus_born, k_exact_dict_bornminus_born\
		 						= KernelComputation(N_v, N_b_minus, N_b, N_k_samples, ZZ_b_minus, Z_b_minus, ZZ_b, Z_b)

		k_bornplus_data, k_exact_bornplus_data, k_dict_bornplus_data, k_exact_dict_bornplus_data\
		  						= KernelComputation(N_v, N_b_plus, N_d, N_k_samples, ZZ_b_plus, Z_b_plus, ZZ_d, Z_d)
		k_bornminus_data, k_exact_bornminus_data, k_dict_bornminus_data, k_exact_dict_bornminus_data \
								= KernelComputation(N_v, N_b_minus, N_d, N_k_samples,  ZZ_b_minus, Z_b_minus, ZZ_d, Z_d)
	else:
		print("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")

	return k_bornplus_born, k_bornminus_born, k_bornplus_data , k_bornminus_data

def MMDGradSampler(N_v,
					data, born_samples, born_samples_plus, born_samples_minus,
					N_k_samples, k_choice):
	N_d 		= data.shape[0]
	N_b 		= born_samples.shape[0]
	N_b_plus 	= born_samples_plus.shape[0]
	N_b_minus	= born_samples_minus.shape[0]

	#Compute the kernel function with inputs required for MMD, [(x,x), (x, y), (y, y)], x samples from Born, y samples from data
	kbplusb, kbminusb, kbplusd , kbminusd  = MMDKernelforGradSampler(N_v, data, born_samples, born_samples_plus,\
																 born_samples_minus, N_k_samples, k_choice)
	#return the gradient of the loss function (L = MMD^2) for a given parameter
	return (2/N_b_minus*N_b)*(kbminusb.sum()) - (2/N_b_plus*N_b)*(kbplusb.sum()) \
			- (2/N_b_minus*N_d)*(kbminusd.sum()) + (2/N_b_plus*N_d)*(kbplusd.sum())

def MMDCost(N_v, data, born_samples, N_kernel_samples, kernel_choice):

	N_d = data.shape[0]
	N_b = born_samples.shape[0]
	'''This function computes the (squared) MMD (Maximum Mean Discrepancy) between the data distribution and the Born machine distribution
	L = MMD^2 = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)], x are samples drawn from the output distribution of the Born Machine, y are
	samples drawn from the data distribution. If we take m samples from the machine, and n from the data, an unbiased estimator of
	the above cost function is: L = 1/m(m-1)sum^m_{i notequal j}k(x_i, x_j)	+1/n(n-1)sum^{n}_{i notequal j}k(y_i, y_j) -2/mn sum^{nm}_{i, j =1}k(x_i, y_j)
	'''
	#Compute the kernel function with inputs required for MMD, [(x,x), (x, y), (y, y)], x samples from Born, y samples from data
	'''Compute kernel given samples from the Born Machine (born_samples) and the Data Distribution (data_samples)
		This must be done for every sample from each distribution (batch gradient descent), (x, y)'''

	if (kernel_choice == 'Gaussian'):
		sigma = np.array([0.25,10,1000])
		k_bd = GaussianKernel(born_samples, data, sigma)
		k_bb = GaussianKernel(born_samples, born_samples, sigma)
		k_dd = GaussianKernel(data, data, sigma)

	elif (kernel_choice == 'Quantum'):
		ZZ_born, Z_born = EncodingFunc(N_v, born_samples, N_b)
		ZZ_data, Z_data = EncodingFunc(N_v, data, N_d)

		k_bd, k_bd_exact, k_dict_bd, k_dict_bd_exact = KernelComputation(N_v, N_b, N_d, N_kernel_samples, ZZ_born, Z_born, ZZ_data, Z_data)
		k_bb, k_bb_exact, k_dict_bb, k_dict_bb_exact = KernelComputation(N_v, N_b, N_b, N_kernel_samples, ZZ_born, Z_born, ZZ_born, Z_born)
		k_dd, k_dd_exact, k_dict_dd, k_dict_dd_exact = KernelComputation(N_v, N_d, N_d, N_kernel_samples, ZZ_data, Z_data, ZZ_data, Z_data)

	else:
		print("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")
	#Compute the loss function (L = MMD^2)
	L = (1/(N_b*(N_b -1)))*((k_bb.sum(axis = 1) - k_bb.diagonal()).sum()) + (1/(N_d*(N_d-1)))*((k_dd.sum(axis = 1) \
		- k_dd.diagonal()).sum()) - (1/(N_b*N_d))*(k_bd.sum())

	print("\nThe MMD loss function is: ", L)
	return L, k_bb, k_dd, k_bd

##############################################################################################################
#If Everything is to be computed exactly
##############################################################################################################
def MMDKernelforGradExact(N_v, bin_visible, N_k_samples, k_choice):
	#If the input corresponding to the kernel choice is either the gaussian kernel or the quantum kernel
	if (k_choice == 'Gaussian'):
		sigma = np.array([0.25, 10, 1000])
		k, k_exact_dict = GaussianKernelExact(N_v, bin_visible, sigma)
		#Gaussian approx kernel is equal to exct kernel
		k_exact = k
		k_dict = k_exact_dict
	elif (k_choice ==  'Quantum'):
		#compute for all binary strings
		ZZ, Z = EncodingFunc(N_v, bin_visible, 2**N_v)
		k, k_exact, k_dict, k_exact_dict = KernelComputation(N_v, 2**N_v , 2**N_v, N_k_samples, ZZ, Z, ZZ, Z)
	else:
		print("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")

	#compute the expectation values for including each binary string sample
	return k, k_exact, k_dict, k_exact_dict

def MMDCostExact(N_v, data_exact_dict, born_probs_dict, kernel_exact_dict):
	N_born_probs = len(born_probs_dict)
	N_data= len(data_exact_dict)

	born_data = np.zeros((N_born_probs, N_data))
	born_born = np.zeros((N_born_probs, N_born_probs))
	data_data = np.zeros((N_data, N_data))
	#compute the term kp_1p_2 to add to expectation value for each pair, from born/bornplus/bornminus machine, and data
	for sampleborn1 in range(0, N_born_probs):
		sampleborn1_string= ConvertToString(sampleborn1, N_v)
		for sampleborn2 in range(0, N_born_probs):
			sampleborn2_string= ConvertToString(sampleborn2, N_v)
			born_born[sampleborn1, sampleborn2] = born_probs_dict[sampleborn1_string]\
									*born_probs_dict[sampleborn2_string]*kernel_exact_dict[(sampleborn1_string, sampleborn2_string)]
		for sampledata in range(0, N_data):
			sampledata_string= ConvertToString(sampledata, N_v)
			born_data[sampleborn1, sampledata] = born_probs_dict[sampleborn1_string]\
							*data_exact_dict[sampledata_string]*kernel_exact_dict[(sampleborn1_string, sampledata_string)]

	for sampledata1 in range(0, N_data):
		sampledata1_string= ConvertToString(sampledata1, N_v)
		for sampledata2 in range(0, N_born_probs):
			sampledata2_string= ConvertToString(sampledata2, N_v)
			data_data[sampledata1, sampledata2] = born_probs_dict[sampledata1_string]\
									*born_probs_dict[sampledata2_string]*kernel_exact_dict[(sampledata1_string, sampledata2_string)]

	#return the loss function (L = MMD^2)
	return born_born.sum() - 2*born_data.sum() + data_data.sum()

def MMDGradExact(N_v,
				data_exact_dict, born_probs_dict, born_probs_plus_dict, born_probs_minus_dict,
				N_k_samples, k_choice):
	#We have dictionaries of 5 files, the kernel, the data, the born probs, the born probs plus/minus
	k_exact_dict = KernelDictFromFile(N_v, N_k_samples, k_choice)

	N_born_probs = len(born_probs_dict)
	N_born_plus_probs = len(born_probs_plus_dict)
	N_born_minus_probs = len(born_probs_minus_dict)
	N_data= len(data_exact_dict)

	data_plus = np.zeros((N_data, N_born_plus_probs))
	data_minus = np.zeros((N_data, N_born_minus_probs))
	born_plus = np.zeros((N_born_probs, N_born_plus_probs))
	born_minus = np.zeros((N_born_probs, N_born_minus_probs))

	#compute the term kp_1p_2 to add to expectation value for each pair, from bornplus/bornminus, with born machine and data
	for sampleborn1 in range(0, N_born_probs):
		sampleborn1_string = ConvertToString(sampleborn1, N_v)
		for samplebornplus in range(0, N_born_plus_probs):
			samplebornplus_string = ConvertToString(samplebornplus, N_v)
			born_plus[sampleborn1, samplebornplus] = born_probs_dict[sampleborn1_string]\
								*born_probs_plus_dict[samplebornplus_string]*k_exact_dict[(sampleborn1_string, samplebornplus_string)]

		for samplebornminus in range(0, N_born_plus_probs):
			samplebornminus_string = ConvertToString(samplebornminus, N_v)
			born_minus[sampleborn1, samplebornminus] = born_probs_dict[sampleborn1_string]\
					*born_probs_minus_dict[samplebornminus_string]*k_exact_dict[(sampleborn1_string, samplebornminus_string)]

	for sampledata1 in range(0, N_data):
		sampledata1_string = ConvertToString(sampledata1, N_v)
		for samplebornplus in range(0, N_born_plus_probs):
			samplebornplus_string = ConvertToString(samplebornplus, N_v)
			data_plus[sampledata1, samplebornplus] = data_exact_dict[sampledata1_string]\
						*born_probs_plus_dict[samplebornplus_string]*k_exact_dict[(sampleborn1_string, samplebornplus_string)]

		for samplebornminus in range(0, N_born_plus_probs):
			samplebornminus_string = ConvertToString(samplebornminus, N_v)
			data_minus[sampledata1, samplebornminus] = data_exact_dict[sampledata1_string]\
						*born_probs_minus_dict[samplebornminus_string]*k_exact_dict[sampleborn1_string, samplebornminus_string]

	#Return the gradient of the loss function (L = MMD^2) for a given parameter
	return 2*(born_plus.sum()-born_minus.sum()-data_minus.sum()+data_plus.sum())

################################################################################################################
#Train Model Using MMD with either exact kernel and gradient or approximate one using samples
################################################################################################################
def MMDTrain(N, N_h, N_v,
			J_i, b_i, g_x_i, g_y_i,
			N_epochs, N_d, N_b, N_b_plus, N_b_minus, N_k_samples,
			data, data_exact_dict,
			k_choice, learning_rate, approx):
	#Initialise a 3 dim array for the graph weights, 2 dim array for biases and gamma parameters
	J = np.zeros((N, N, N_epochs))
	b = np.zeros((N,  N_epochs))
	gamma_x = np.zeros((N,  N_epochs))
	gamma_y = np.zeros((N,  N_epochs))

	#Import initial parameter values
	J[:,:,0] = J_i
	b[:,0] =b_i
	gamma_x[:,0] = g_x_i
	gamma_y[:,0] = g_y_i

	#gamma_x/gamma_y is not to be trained, set gamma values to be constant at each epoch
	for epoch in range(0, N_epochs-1):
		gamma_x[:,epoch + 1] = gamma_x[:, epoch]
		gamma_y[:,epoch + 1] = gamma_y[:, epoch]

	#Initialise the gradient arrays, each element is one parameter
	bias_grad = np.zeros((N))
	weight_grad = np.zeros((N, N))
	L = np.zeros((N_epochs-1))

	for epoch in range(0, N_epochs-1):
		print("\nThis is Epoch number: ", epoch)
		Jt = J[:,:,epoch]
		bt = b[:,epoch]
		gxt = gamma_x[:,epoch]
		gyt = gamma_y[:,epoch]

		#generate samples, and exact probabilities for current set of parameters
		born_samples, born_probs_dict = BornSampler(N, N_v, N_h, N_b, Jt, bt, gxt, gyt)

		'''Updating bias b[r], control set to 'BIAS' '''
		for bias_index in range(0,N):
			born_samples_plus, born_samples_minus, born_probs_dict_plus, born_probs_dict_minus = PlusMinusSampleGen(N, N_v, N_h, \
												Jt, bt, gxt, gyt, \
												0,0, bias_index, \
												'QAOA', 'BIAS',\
												 N_b_plus, N_b_minus)
			##If the exact MMD is to be computed approx == 'Exact', if only approximate version using samples, approx == 'Sampler'
			print(approx)
			if approx == 'Sampler':
				bias_grad[bias_index] = MMDGradSampler(N_v,\
				 											data, born_samples, born_samples_plus, born_samples_minus,\
															N_k_samples, k_choice)
			elif approx == 'Exact':
				bias_grad[bias_index] = MMDGradExact(N_v,\
				 									data_exact_dict, born_probs_dict, \
													born_probs_dict_plus, born_probs_dict_minus,\
													'infinite', k_choice)

		b[:, epoch + 1] = b[:, epoch] - learning_rate*bias_grad

		'''Updating weight J[p,q], control set to 'WEIGHTS' '''
		for j in range(0, N):
			i = 0
			while(i < j):

				born_samples_plus, born_samples_minus, born_probs_dict_plus, born_probs_dict_minus \
								= PlusMinusSampleGen(N, N_v, N_h,\
								 					Jt, bt, gxt, gyt, \
													i, j , 0,\
													'QAOA', 'WEIGHTS',\
													 N_b_plus, N_b_minus)
				##If the exact MMD is to be computed approx == 'Exact', if only approximate version using samples, approx == 'Sampler'
				if approx == 'Sampler':
					weight_grad[i,j] = MMDGradSampler(N_v, data, born_samples, born_samples_plus, born_samples_minus, N_k_samples, k_choice)
				elif approx == 'Exact':
					#N_kernel_samples = 'infinite' since exact gradient is being computed
					weight_grad[i,j] = MMDGradExact(N_v, \
													data_exact_dict, born_probs_dict, \
													born_probs_dict_plus, born_probs_dict_minus,\
													'infinite', k_choice)
				i = i+1
		J[:,:, epoch+1] = J[:,:, epoch] - learning_rate*(weight_grad + np.transpose(weight_grad))

		'''Check MMD of model distribution'''
		if approx == 'Sampler':
			L[epoch], kbb, kdd, kbd = MMDCost(N_v, data, born_samples, N_k_samples, k_choice)
		elif approx == 'Exact':
			k_exact_dict = KernelDictFromFile(N_v, 'infinite', k_choice)
			L[epoch] = MMDCostExact(N_v, data_exact_dict, born_probs_dict, k_exact_dict)
		print("The MMD Loss for epoch ", epoch, "is", L[epoch])
	return L, J, b, gamma_x, gamma_y
