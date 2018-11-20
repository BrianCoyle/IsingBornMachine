
import numpy as np

from param_init import StateInit, NetworkParams

from classical_kernel import GaussianKernel, GaussianKernelExact
from file_operations_in import KernelDictFromFile, DataDictFromFile
from mmd_kernel import  KernelComputation, EncodingFunc
from auxiliary_functions import SampleArrayToList, ConvertToString, EmpiricalDist
import stein_score as ss
import json

def SumContribution(N_qubits, dict_one, dict_two, kernel_dict):
	'''
	Computes the contribution to the MMD/MMD gradient for two sets of probabilities, dict_one 
	and dict_two'''
	N_probs_one = len(dict_one)
	N_probs_two = len(dict_two)

	first_second = np.zeros((N_probs_one, N_probs_one))

	for first_term in range(0, N_probs_one):
		first_string = ConvertToString(first_term, N_qubits)
		for second_term in range(0, N_probs_two):
			second_string = ConvertToString(second_term, N_qubits)
			first_second[first_term, second_term] = dict_one[first_string]\
										*dict_two[second_string]*kernel_dict[(first_string, second_string)]

	return first_second
	
def KernelSum(N_qubits, samplearray1, samplearray2,
			kernel_sampled_dict, *argsv):
	'''This function computes the contribution to the MMD from the empirical distibutions
			from two sets of samples'''

	N_sample1 = samplearray1.shape[0]
	N_sample2 = samplearray2.shape[0]

	empiricaldist_dict1 = EmpiricalDist(samplearray1, N_qubits)
	empiricaldist_dict2 = EmpiricalDist(samplearray2, N_qubits)

	samples_list_1 = SampleArrayToList(samplearray1)
	samples_list_2 = SampleArrayToList(samplearray2)

	kernel_sum_unnorm = 0
	for sample_index1 in range(0, N_sample1):
		sample1 = samples_list_1[sample_index1]
		for sample_index2 in range(0, N_sample2):
			sample2 = samples_list_2[sample_index2]
		
			if (sample1 == sample2):
				if ('same' not in argsv):
					kernel_sum_unnorm = kernel_sum_unnorm + kernel_sampled_dict[sample1, sample2]
			else:
				kernel_sum_unnorm = kernel_sum_unnorm + kernel_sampled_dict[sample1, sample2]
	
	if ('same' not in argsv):
		kernel_sum = (1/(N_sample1*N_sample2))*kernel_sum_unnorm
	else: 
		kernel_sum = (1/(N_sample1*(N_sample2-1)))*kernel_sum_unnorm

	kernel_prob = SumContribution(N_qubits, empiricaldist_dict1 ,empiricaldist_dict2, kernel_sampled_dict)


	return kernel_sum, kernel_prob

def SteinGrad(N_qubits, data, data_exact_dict,
						born_samples, born_probs_dict,
						born_samples_plus, born_plus_exact_dict, 
						born_samples_minus, born_minus_exact_dict,
						N_samples, k_choice, approx, score_approx, chi, stein_kernel_choice):

	data_samples_list= SampleArrayToList(data)
	born_samples_list= SampleArrayToList(born_samples)
	bornplus_samples_list= SampleArrayToList(born_samples_plus)
	bornminus_samples_list= SampleArrayToList(born_samples_minus)

	kernel_dict_for_stein  = KernelDictFromFile(N_qubits, N_samples, stein_kernel_choice)

	emp_born_dict = EmpiricalDist(born_samples, N_qubits)
	emp_born_plus_dict = EmpiricalDist(born_samples_plus, N_qubits)
	emp_born_minus_dict = EmpiricalDist(born_samples_minus, N_qubits)

	expectation_value_summand_1 = {}
	expectation_value_summand_2 = {}
	expectation_value_summand_3 = {}
	expectation_value_summand_4 = {}
	#Compute the weighted kernel for each pair of samples required in the gradient of Stein Cost Function
	kappa_q_born_bornplus = ss.ComputeWeightedKernel(N_qubits, kernel_dict_for_stein, data_samples_list,\
													data_exact_dict, born_samples_list, bornplus_samples_list, score_approx, chi)
	kappa_q_bornplus_born = ss.ComputeWeightedKernel(N_qubits, kernel_dict_for_stein, data_samples_list, \
													data_exact_dict, bornplus_samples_list,born_samples_list, score_approx, chi)
	kappa_q_born_bornminus = ss.ComputeWeightedKernel(N_qubits, kernel_dict_for_stein, data_samples_list,\
													data_exact_dict, born_samples_list, bornminus_samples_list, score_approx, chi)
	kappa_q_bornminus_born = ss.ComputeWeightedKernel(N_qubits, kernel_dict_for_stein, data_samples_list,\
													data_exact_dict, bornminus_samples_list, born_samples_list, score_approx, chi)
	

	for sample1 in bornminus_samples_list:
		for sample2 in born_samples_list:
	
			if (sample1, sample2) in kappa_q_born_bornminus:
				expectation_value_summand_1[(sample1, sample2)] = emp_born_minus_dict[sample1]*kappa_q_bornminus_born[(sample1, sample2)]*emp_born_dict[sample2]

			if (sample1, sample2) in kappa_q_born_bornminus:
				expectation_value_summand_3[(sample1, sample2)] = emp_born_dict[sample1]*kappa_q_born_bornminus[(sample1, sample2)]*emp_born_minus_dict[sample2]

	for sample1 in bornplus_samples_list:
		for sample2 in born_samples_list:
			if (sample1, sample2) in kappa_q_bornplus_born:
				expectation_value_summand_2[(sample1, sample2)] = emp_born_plus_dict[sample1]*kappa_q_bornplus_born[(sample1, sample2)]*emp_born_dict[sample2]
				
			if (sample1, sample2) in kappa_q_born_bornplus:
				expectation_value_summand_4[(sample1, sample2)] = emp_born_dict[sample1]*kappa_q_born_bornplus[(sample1, sample2)]*emp_born_plus_dict[sample2]


	# L_stein_grad =  expectation_value_summand_1.sum() - expectation_value_summand_2.sum() +expectation_value_summand_3.sum()-expectation_value_summand_3.sum()
	L_stein_grad =  sum(list(expectation_value_summand_1.values()))- sum(list(expectation_value_summand_2.values())) +  \
					sum(list(expectation_value_summand_3.values()))- sum(list(expectation_value_summand_4.values()))

	# if (approx == 'Sampler'):

	# 	kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_samples[4], k_choice)

	# 	k_bornplus_born, bornplus_born = KernelSum(N_qubits, born_samples, born_samples_plus,\
	# 													kernel_sampled_dict)
	# 	k_bornminus_born, bornminus_born = KernelSum(N_qubits, born_samples, born_samples_minus,\
	# 													kernel_sampled_dict)
	# 	k_bornplus_data,  bornplus_data = KernelSum(N_qubits, data, born_samples_plus,\
	# 													kernel_sampled_dict)
	# 	k_bornminus_data,  bornminus_data = KernelSum(N_qubits, data, born_samples_minus,\
	# 													kernel_sampled_dict)
		
		
	# 	#L1_MMD_Grad = 2*(k_bornminus_born - k_bornplus_born- k_bornminus_data + k_bornplus_data)
	# 	L_MMD_Grad = 2*(bornplus_born.sum()-bornminus_born.sum()-bornminus_data.sum()+bornplus_data.sum())

	# elif (approx == 'Exact'):
	# 	#We have dictionaries of 5 files, the kernel, the data, the born probs, the born probs plus/minus
	# 	kernel_dict = KernelDictFromFile(N_qubits, N_samples[4], k_choice)
	# 	#compute the term kp_1p_2 to add to expectation value for each pair, from bornplus/bornminus, with born machine and data
	# 	born_born_plus = SumContribution(N_qubits, born_probs_dict, born_plus_exact_dict, kernel_dict)
	# 	born_born_minus = SumContribution(N_qubits, born_probs_dict, born_minus_exact_dict, kernel_dict)
	# 	data_born_plus = SumContribution(N_qubits, data_exact_dict, born_plus_exact_dict, kernel_dict)
	# 	data_born_minus = SumContribution(N_qubits, data_exact_dict, born_minus_exact_dict, kernel_dict)

	# 	#Return the gradient of the loss function (L = MMD^2) for a given parameter
	# 	L_MMD_Grad = 2*(born_born_plus.sum()-born_born_minus.sum()-data_born_minus.sum()+data_born_plus.sum())
	return L_stein_grad



def SteinCost(N_qubits, data_samples, data_exact_dict, born_samples, born_probs_dict,	
			N_kernel_samples, kernel_choice,
			approx, score_approx, chi, stein_kernel_choice):
	'''This function computes the MMD cost function between P and Q from samples from P and
		Q if the sampling approx is taken, otherwise if the cost function is to be computed exactly'''

	data_samples_list= SampleArrayToList(data_samples)
	born_samples_list= SampleArrayToList(born_samples)
	

	if (approx == 'Sampler'):
		#Compute the empirical data distibution given samples
	
		kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_kernel_samples, kernel_choice)
		data_samples_list = SampleArrayToList(data_samples)
		emp_born_dist = EmpiricalDist(born_samples, N_qubits)

		# expectation_value_summand = np.zeros((len(born_samples)))
		expectation_value_summand = {}

		kappa_q = ss.ComputeWeightedKernel(N_qubits, kernel_sampled_dict, data_samples_list, data_exact_dict, born_samples_list, born_samples_list, score_approx, chi)
		for sample1 in born_samples_list:
			for sample2 in born_samples_list:
				if (sample1, sample2) in kappa_q:
						expectation_value_summand[(sample1, sample2)] = emp_born_dist[sample1]*kappa_q[(sample1, sample2)]*emp_born_dist[sample2]

		L =   sum(list(expectation_value_summand.values()))

	elif (approx == 'Exact'):

		#compute the term kp_1p_2 to add to expectation value for each pair, from born/bornplus/bornminus machine, and data
		k_exact_dict = KernelDictFromFile(N_qubits, 'infinite', kernel_choice)

		born_born = SumContribution(N_qubits, born_probs_dict, born_probs_dict, k_exact_dict)
		born_data = SumContribution(N_qubits, data_exact_dict, born_probs_dict, k_exact_dict)
		data_data = SumContribution(N_qubits, data_exact_dict, data_exact_dict, k_exact_dict)
		
		#return the loss function (L = MMD^2)
		L = born_born.sum() - 2*born_data.sum() + data_data.sum()

	return L


def SteinKernel(N_qubits, bin_visible, N_samples, k_choice):
	#If the input corresponding to the kernel choice is either the gaussian kernel or the quantum kernel
	if (k_choice == 'Gaussian'):
		sigma = np.array([0.25, 10, 1000])
		k, k_exact_dict = GaussianKernelExact(N_qubits, bin_visible, sigma)
		#Gaussian approx kernel is equal to exact kernel
		k_exact = k
		k_dict = k_exact_dict
	elif (k_choice ==  'Quantum'):
		#compute for all binary strings
		ZZ, Z = EncodingFunc(N_qubits, bin_visible)
		k, k_exact, k_dict, k_exact_dict = KernelComputation(N_qubits, 2**N_qubits , 2**N_qubits, N_samples[4], ZZ, Z, ZZ, Z)
	else:
		print("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")

	#compute the expectation values for including each binary string sample
	return k, k_exact, k_dict, k_exact_dict


