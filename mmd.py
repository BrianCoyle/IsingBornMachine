from pyquil.quil import Program
import numpy as np
from pyquil.api import get_qc

from auxiliary_functions import ConvertToString, EmpiricalDist
from param_init import StateInit, NetworkParams
from sample_gen import BornSampler, PlusMinusSampleGen

from train_generation import TrainingData, DataSampler
from classical_kernel import GaussianKernel, GaussianKernelExact
from file_operations_in import KernelDictFromFile
from mmd_kernel import KernelCircuit, KernelComputation, EncodingFunc
from mmd_sampler2 import MMDGrad, MMDCost, MMDKernel


################################################################################################################
#Train Model Using MMD with either exact kernel and gradient or approximate one using samples
################################################################################################################
def MMDTrain(N, N_h, N_v,
			J_i, b_i, g_x_i, g_y_i,
			N_epochs, N_d, N_b, N_k_samples,
			data, data_exact_dict,
			k_choice, learning_rate, approx, weight_sign):
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

	#Initialise the gradient arrays, each element is one parameter
	bias_grad = np.zeros((N))
	gamma_x_grad = np.zeros((N))

	weight_grad = np.zeros((N, N))
	L = np.zeros((N_epochs-1))
	born_probs_list = []
	empirical_probs_list = []

	circuit_choice ='QAOA'
	
	for epoch in range(0, N_epochs-1):
		#gamma_x/gamma_y is not to be trained, set gamma values to be constant at each epoch
		gamma_y[:,epoch + 1] = gamma_y[:, epoch]
		gamma_x[:,epoch + 1] = gamma_x[:, epoch]


		print("\nThis is Epoch number: ", epoch)

		Jt = J[:,:,epoch]
		bt = b[:,epoch]
		gxt = gamma_x[:,epoch]
		gyt = gamma_y[:,epoch]
		#generate samples, and exact probabilities for current set of parameters
		born_samples, born_probs_dict = BornSampler(N, N_v, N_h, N_b, Jt, bt, gxt, gyt, circuit_choice)
		#Flip the samples to be consistent with exact probabilities derived from simulator [00010] -> [01000]
		born_probs_list.append(born_probs_dict)
		
		empirical_probs_dict = EmpiricalDist(born_samples, N_v)
		empirical_probs_list.append(empirical_probs_dict)

		print('The Born Machine Outputs Probabilites\n', born_probs_dict)
		print('The empirical Born is:\n', empirical_probs_dict)
		print('The Data is\n,', data_exact_dict)

		'''Updating bias b[r], control set to 'BIAS' '''
		for bias_index in range(0,N):
			born_samples_plus, born_samples_minus, born_plus_exact_dict, born_minus_exact_dict\
								 = PlusMinusSampleGen(N, N_v, N_h, \
												Jt, bt, gxt, gyt, \
												0,0, bias_index, 0,\
												circuit_choice, 'BIAS',\
												N_b, N_b)
			
			##If the exact MMD is to be computed approx == 'Exact', if only approximate version using samples, approx == 'Sampler'
		
			bias_grad[bias_index] = MMDGrad(N_v, data, data_exact_dict,
						born_samples, born_probs_dict,\
						born_samples_plus, born_plus_exact_dict, \
						born_samples_minus, born_minus_exact_dict,\
						N_k_samples, k_choice, approx)

		b[:, epoch + 1] = b[:, epoch] + weight_sign*learning_rate*bias_grad
		print('bias_grad is:', bias_grad)

		# if (circuit_choice == 'QAOA'):	
		# 	'''Updating finalparam gamma[s], control set to 'FINALPARAM' '''
		# 	for gammaindex in range(0,N):
		# 		born_samples_plus, born_samples_minus, born_plus_exact_dict, born_minus_exact_dict\
		# 							= PlusMinusSampleGen(N, N_v, N_h, \
		# 											Jt, bt, gxt, gyt, \
		# 											0, 0, 0, gammaindex, \
		# 											circuit_choice, 'GAMMA',\
		# 											N_b, N_b)
		# 		
			
		# 		##If the exact MMD is to be computed approx == 'Exact', if only approximate version using samples, approx == 'Sampler'
		# 		gamma_x_grad[gammaindex] =MMDGrad(N_v, data, data_exact_dict,
		# 										born_samples, born_probs_dict,\
		# 										born_samples_plus, born_plus_exact_dict, \
		# 										born_samples_minus, born_minus_exact_dict,\
		# 										N_k_samples, k_choice, approx)

		# 	print('gamma_x_grad is:', gamma_x_grad)

		# 	gamma_x[:, epoch + 1] = gamma_x[:, epoch] + weight_sign*learning_rate*gamma_x_grad
		'''Updating weight J[p,q], control set to 'WEIGHTS' '''
		for q in range(0, N):
			p = 0
			while(p < q):
				## Draw samples from +/- pi/2 shifted circuits for each weight update, J_{p, q}
				born_samples_plus, born_samples_minus, born_plus_exact_dict, born_minus_exact_dict \
								= PlusMinusSampleGen(N, N_v, N_h,\
								 					Jt, bt, gxt, gyt, \
													p, q , 0, 0,\
													circuit_choice, 'WEIGHTS',\
													 N_b, N_b)
	

				##If the exact MMD is to be computed approx == 'Exact', if only approximate version using samples, approx == 'Sampler'
				# if approx == 'Sampler':
				weight_grad[p,q] = MMDGrad(N_v, data, data_exact_dict,
										born_samples, born_probs_dict,\
										born_samples_plus, born_plus_exact_dict, \
										born_samples_minus, born_minus_exact_dict,\
										N_k_samples, k_choice, approx)

				p = p+1
		J[:,:, epoch+1] = J[:,:, epoch] + weight_sign*learning_rate*(weight_grad + np.transpose(weight_grad))
		'''Check MMD of model distribution'''

		L[epoch], kbb, kbd, kdd = MMDCost(N_v, data, data_exact_dict, born_samples,\
											 born_probs_dict, N_k_samples, k_choice, approx)

		print("The MMD Loss for epoch ", epoch, "is", L[epoch])
		



	return L, J, b, gamma_x, gamma_y, born_probs_list, empirical_probs_list

