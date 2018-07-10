from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
import numpy as np
from numpy import pi,log2
from pyquil.api import QVMConnection
from random import *
from pyquil.quilbase import DefGate
from pyquil.parameters import Parameter, quil_exp, quil_cos, quil_sin
from Param_Init import state_init, network_params

from Sample_Gen import born_sampler, plusminus_sample_gen
from MMD_Kernel_Comp import kernel_computation, encoding_func, mmd_cost
from Train_Generation import training_data, data_sampler
from Classical_Kernel import gaussian_kernel

def mmd_grad_kernel_comp(N_v, data, born_samples, born_samples_plus, born_samples_minus, N_kernel_samples, kernel_choice):
	N_d 		= data.shape[0]
	N_b 		= born_samples.shape[0]
	N_b_plus 	= born_samples_plus.shape[0]
	N_b_minus	= born_samples_minus.shape[0]

	if (kernel_choice == 'gaussian' or 'Gaussian'):
		#If the input corresponding to the kernel choice is either the gaussian kernel or the quantum kernel
		sigma = np.array([10,100,1000])
		
		kernel_bornplus_born = gaussian_kernel(born_samples_plus, born_samples, sigma)
		kernel_bornminus_born = gaussian_kernel(born_samples_minus, born_samples, sigma)
		kernel_bornplus_data = gaussian_kernel(born_samples_plus, data, sigma)
		kernel_bornminus_data = gaussian_kernel(born_samples_minus, data, sigma)

	elif (kernel_choice == 'quantum kernel' or 'Quantum Kernel'):
	
		ZZ_b, Z_b = encoding_func(N_v, born_samples, N_b)

		ZZ_b_plus, Z_b_plus = encoding_func(N_v, born_samples_plus, N_b_plus)

		ZZ_b_minus, Z_b_minus = encoding_func(N_v, born_samples_minus, N_b_minus)

		ZZ_d, Z_d = encoding_func(N_v, data, N_d)

		kernel_bornplus_born = kernel_computation(N_v, N_b_plus, N_b, N_kernel_samples, ZZ_b_plus, Z_b_plus, ZZ_b, Z_b)
		kernel_bornminus_born = kernel_computation(N_v, N_b_minus, N_b, N_kernel_samples, ZZ_b_minus, Z_b_minus, ZZ_b, Z_b)

		kernel_bornplus_data = kernel_computation(N_v, N_b_plus, N_d, N_kernel_samples, ZZ_b_plus, Z_b_plus, ZZ_d, Z_d)
		kernel_bornminus_data = kernel_computation(N_v, N_b_minus, N_d, N_kernel_samples,  ZZ_b_minus, Z_b_minus, ZZ_d, Z_d)
	else:
		print("Please enter either 'Gaussian' or 'Quantum Kernel' to choose a kernel")

	return kernel_bornplus_born, kernel_bornminus_born, kernel_bornplus_data , kernel_bornminus_data 


def mmd_grad_comp(N_v, data, born_samples, born_samples_plus, born_samples_minus, N_kernel_samples, kernel_choice):

	N_d 		= data.shape[0]
	N_b 		= born_samples.shape[0]
	N_b_plus 	= born_samples_plus.shape[0]
	N_b_minus	= born_samples_minus.shape[0]

	'''[FIX ]      This function computes the gradient of the (squared) MMD (Maximum Mean Discrepancy) between the data distribution and the Born machine distribution
	L = MMD^2 = 2E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)], x are samples drawn from the output distribution of the Born Machine, y are 
	samples drawn from the data distribution. If we take m samples from the machine, and n from the data, an unbiased estimator of 
	the above cost function is: L = 1/m(m-1)sum^m_{i notequal j}k(x_i, x_j)	+1/n(n-1)sum^{n}_{i notequal j}k(y_i, y_j) -2/mn sum^{nm}_{i, j =1}k(x_i, y_j)
	'''

	#Compute the kernel function with inputs required for MMD, [(x,x), (x, y), (y, y)], x samples from Born, y samples from data
	kbplusb, kbminusb, kbplusd , kbminusd  = mmd_grad_kernel_comp(N_v, data, born_samples, born_samples_plus,\
																 born_samples_minus, N_kernel_samples, kernel_choice)

	#Compute the gradient of the loss function (L = MMD^2) for a given parameter

	gradient = (2/N_b_plus*N_b)*(kbplusb.sum()) - (2/N_b_minus*N_b)*(kbminusb.sum()) \
					- (2/N_b_plus*N_d)*(kbplusd.sum()) + (2/N_b_minus*N_d)*(kbminusd.sum())
	
	return gradient

def mmd_train(N, N_h, N_v, N_epochs, N_d, N_b, N_b_plus, N_b_minus, N_kernel_samples, data, kernel_choice):

	#Initialise a 3 dim array for the graph weights, 2 dim array for biases and gamma parameters

	J = np.zeros((N, N, N_epochs))
	b = np.zeros((N,  N_epochs))
	gamma_x = np.zeros((N,  N_epochs))
	gamma_y = np.zeros((N,  N_epochs))

	#Set learning rate for parameter updates
	learning_rate = 0.1
	
	#Initialize Parameters, J, b for epoch 0 at random, gamma = constant = pi/4
	J[:,:,0], b[:,0], gamma_x[:,0], gamma_y[:,0] = network_params(N, J[:,:,0], b[:,0], gamma_x[:,0], gamma_y[:,0])

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
		
		#generate Samples for current set of parameters

		
		born_samples = born_sampler(N, N_v, N_h, N_b, J[:,:,epoch],
									 b[:,epoch], gamma_x[:,epoch], gamma_y[:,epoch])

		#Compute output distribution for current set of parameters, J, b, gamma_x
		#P_v[:, epoch] = compute_dist(N, N_v, N_h, J[:,:,epoch], b[:,epoch], gamma_x[:, epoch], bin_visible, bin_hidden)

		'''Updating bias b[r], control set to 0'''
		for bias_index in range(0,N):	

			born_samples_plus, born_samples_minus = plusminus_sample_gen(N, N_v, N_h, Jt, bt, gxt, gyt, 0,0,\
																			 bias_index, 1, 0, N_b_plus, N_b_minus)

			bias_grad[bias_index] = mmd_grad_comp(N_v, data, born_samples, born_samples_plus, \
													born_samples_minus, N_kernel_samples, kernel_choice)
			
		print('bias_grad for epoch',  epoch, 'is: ', bias_grad)

		b[:, epoch + 1] = b[:, epoch] - learning_rate*bias_grad

		
		'''Updating weight J[p,q], control set to 1'''
		for j in range(0, N):
			i = 0
			while(i < j):

				born_samples_plus, born_samples_minus = plusminus_sample_gen(N, N_v, N_h, Jt, bt, gxt,\
								gyt, i,j , 0, 1, 1, N_b_plus, N_b_minus)

				weight_grad[i,j] = mmd_grad_comp(N_v, data, born_samples, born_samples_plus, \
																born_samples_minus, N_kernel_samples, kernel_choice)
				
				i = i+1
		print('weight_grad for epoch',  epoch, 'is: ', weight_grad)

		J[:,:, epoch+1] = J[:,:, epoch] - learning_rate*(weight_grad + np.transpose(weight_grad))


		#print('delta_weight for epoch',  epoch, 'is: ', delta_weight)

		#print('J for epoch ',  epoch, 'is: ', b[:, epoch])
		# print('delta_bias for ', bias_index, 'is: ', delta_bias[bias_index])
		#print('\n J', epoch, '= ',J[:,:,epoch], '\n J',epoch+1, '= ', J[:,:,epoch+1])
		
		#wavefunction, probs = state_init(N, N_v, N_h, J[:,:,epoch], b[:,:,epoch],gamma[:,:,epoch])
		#P_v[:, epoch] = compute_dist(N, N_v, N_h, J[:,:, epoch], b[:,epoch], gamma[:, epoch], i, j, 0, bin_visible, bin_hidden, 1)
		'''Check MMD Divergence of model distribution'''

		L[epoch], kbb, kdd, kbd = mmd_cost(N_v, data, born_samples, N_kernel_samples, kernel_choice)

	return L, J, b, gamma_x, gamma_y








