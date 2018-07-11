from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
import numpy as np
from pyquil.api import QVMConnection

from Train_Generation import training_data, data_sampler
from Param_Init import network_params, state_init

from Sample_Gen import born_sampler
from Kernel_Circuit import kernel_circuit
from Classical_Kernel import gaussian_kernel

'''Define Non-Linear function to encode samples in graph weights/biases'''
#Fill graph weight and bias arrays according to one single sample
def encoding_func(N_v, sample, N_samples):

	ZZ = np.zeros((N_v, N_v, N_samples))
	Z = np.zeros((N_v, N_samples))

	for sample_number in range(0, N_samples):
		for i in range(0, N_v):
			if int(sample[sample_number, i]) == 1:
				Z[i, sample_number] = np.pi
			j = 0
			while (j < i):
				if int(sample[sample_number, i]) == 1 and int(sample[sample_number, j]) == 1:
					ZZ[i,j, sample_number] = pi
					ZZ[j,i, sample_number] = ZZ[i,j, sample_number]
				j = j+1

	return ZZ, Z

def kernel_computation(N_v, N_samples1, N_samples2, N_kernel_samples, ZZ_1, Z_1, ZZ_2, Z_2):

	kernel = np.zeros((N_samples1, N_samples2))

	for i in range(0, N_samples1):
		for j in range(0, N_samples2):

			qvm = QVMConnection()
			prog = kernel_circuit( ZZ_1[:,:,i], Z_1[:,i], ZZ_2[:,:,j], Z_2[:,j], N_v)

			#Index list for classical registers we want to put measurement outcomes into.
			#Measure the kernel circuit to compute the kernel, the kernel is the probability of getting (00...000) outcome.

			classical_regs = list(range(0, N_v))

			for qubit_index in range(0, N_v):
				prog.measure(qubit_index, qubit_index)

			kernel_measurements = np.asarray(qvm.run(prog, classical_regs, N_kernel_samples))
			(m,n) = kernel_measurements.shape

			N_zero_strings = m - np.count_nonzero(np.count_nonzero(kernel_measurements, 1))

			#The kernel is given by = [Number of times outcome (00...000) occurred]/[Total number of measurement runs]
			kernel[i,j] = N_zero_strings/N_kernel_samples

	return kernel


def mmd_cost(N_v, data, born_samples, N_kernel_samples, kernel_choice):
	#print("The data samples are:\n ", data)
	#print("The born samples are:\n ", born_samples)
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

	if (kernel_choice == 'gaussian' or 'Gaussian'):
		sigma = np.array([0.25,10,1000])
		k_bd = gaussian_kernel(born_samples, data, sigma)
		k_bb = gaussian_kernel(born_samples, born_samples, sigma)
		k_dd = gaussian_kernel(data, data, sigma)

	elif (kernel_choice == 'quantum kernel' or 'Quantum Kernel'):
		ZZ_born, Z_born = encoding_func(N_v, born_samples, N_b)
		ZZ_data, Z_data = encoding_func(N_v, data, N_d)

		k_bd = kernel_computation(N_v, N_b, N_d, N_kernel_samples, ZZ_born, Z_born, ZZ_data, Z_data)
		k_bb = kernel_computation(N_v, N_b, N_b, N_kernel_samples, ZZ_born, Z_born, ZZ_born, Z_born)
		k_dd = kernel_computation(N_v, N_d, N_d, N_kernel_samples, ZZ_data, Z_data, ZZ_data, Z_data)

	else:
		print("Please enter either 'Gaussian' or 'Quantum Kernel' to choose a kernel")

	#print("kernel_born_born is ", k_bb)
	#print("kernel_data_data is ", k_dd)
	#print("kernel_born_data is ", k_bd)

	#Compute the loss function (L = MMD^2)
	L = (1/(N_b*(N_b -1)))*((k_bb.sum(axis = 1) - k_bb.diagonal()).sum()) + (1/(N_d*(N_d-1)))*((k_dd.sum(axis = 1) \
		- k_dd.diagonal()).sum()) - (1/(N_b*N_d))*(k_bd.sum())

	print("\nThe MMD loss function is: ", L)

	return L, k_bb, k_dd, k_bd
