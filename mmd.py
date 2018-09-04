from pyquil.quil import Program
import numpy as np
from pyquil.api import QVMConnection

from param_init import StateInit, NetworkParams
from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler, ConvertToString
from classical_kernel import GaussianKernel, GaussianKernelExact
from file_operations_in import KernelDictFromFile
from mmd_kernel import KernelCircuit, KernelComputation, EncodingFunc
from mmd_sampler import MMDKernelforGradSampler,MMDGradSampler, MMDCostSampler
from mmd_exact import MMDKernelExact, MMDGradExact, MMDCostExact


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
		for q in range(0, N):
			p = 0
			while(i < j):
				born_samples_plus, born_samples_minus, born_probs_dict_plus, born_probs_dict_minus \
								= PlusMinusSampleGen(N, N_v, N_h,\
								 					Jt, bt, gxt, gyt, \
													p, q , 0,\
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
			L[epoch], kbb, kdd, kbd = MMDCostSampler(N_v, data, born_samples, N_k_samples, k_choice)
		elif approx == 'Exact':
			k_exact_dict = KernelDictFromFile(N_v, 'infinite', k_choice)
			L[epoch] = MMDCostExact(N_v, data_exact_dict, born_probs_dict, k_exact_dict)
		print("The MMD Loss for epoch ", epoch, "is", L[epoch])
	return L, J, b, gamma_x, gamma_y
