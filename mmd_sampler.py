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
from file_operations_in import KernelDictFromFile
from mmd_kernel import KernelCircuit, KernelComputation, EncodingFunc
import sys
import json

##############################################################################################################
#If Everything is to be computed using samples
##############################################################################################################
def MMDKernelforGradSampler(N_v, data, born_samples,
						born_samples_plus, born_samples_minus,
						N_k_samples, k_choice):

	if (k_choice == 'Gaussian'):
		sigma = np.array([0.25, 10, 1000])
		k_bornplus_born = GaussianKernel(born_samples_plus, born_samples, sigma)
		k_bornminus_born = GaussianKernel(born_samples_minus, born_samples, sigma)
		k_bornplus_data = GaussianKernel(born_samples_plus, data, sigma)
		k_bornminus_data = GaussianKernel(born_samples_minus, data, sigma)
	elif (k_choice == 'Quantum'):
		ZZ_b, Z_b = EncodingFunc(N_v, born_samples)
		ZZ_b_plus, Z_b_plus = EncodingFunc(N_v, born_samples_plus)
		ZZ_b_minus, Z_b_minus = EncodingFunc(N_v, born_samples_minus)
		ZZ_d, Z_d = EncodingFunc(N_v, data)

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

def MMDCostSampler(N_v, data, born_samples, N_kernel_samples, kernel_choice):

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
		ZZ_born, Z_born = EncodingFunc(N_v, born_samples)
		ZZ_data, Z_data = EncodingFunc(N_v, data)

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
