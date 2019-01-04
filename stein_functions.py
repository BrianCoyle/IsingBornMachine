
import numpy as np

from pyquil.api import get_qc
from classical_kernel import GaussianKernelArray, NormaliseKernel
from file_operations_in import KernelDictFromFile, DataDictFromFile
from quantum_kernel import  QuantumKernelComputation
from auxiliary_functions import SampleArrayToList, ConvertToString, EmpiricalDist
import stein_score as ss
import json	

def SteinGrad(device_params, data, data_exact_dict,
				born_samples, born_probs_dict,
				born_samples_plus,  
				born_samples_minus, 
				N_samples, k_choice, stein_params):


	device_name = device_params[0]
	as_qvm_value = device_params[1]

	qc = get_qc(device_name, as_qvm = as_qvm_value)
	qubits = qc.qubits()
	N_qubits = len(qubits)
	
	data_samples_list= SampleArrayToList(data)
	born_samples_list= SampleArrayToList(born_samples)
	bornplus_samples_list= SampleArrayToList(born_samples_plus)
	bornminus_samples_list= SampleArrayToList(born_samples_minus)

	N_born_samples = len(born_samples_list)
	N_bornplus_samples = len(bornplus_samples_list)
	N_bornminus_samples = len(bornminus_samples_list)

	stein_kernel_choice = stein_params[3]
	kernel_dict_for_stein  = KernelDictFromFile(N_qubits, N_samples, stein_kernel_choice)

	# Compute the weighted kernel for each pair of samples required in the gradient of Stein Cost Function
	kappa_q_born_bornplus = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data_samples_list,\
													data_exact_dict, born_samples_list, bornplus_samples_list, stein_params)
	kappa_q_bornplus_born = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data_samples_list, \
													data_exact_dict, bornplus_samples_list,born_samples_list, stein_params)
	kappa_q_born_bornminus = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data_samples_list,\
													data_exact_dict, born_samples_list, bornminus_samples_list, stein_params)
	kappa_q_bornminus_born = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data_samples_list,\
													data_exact_dict, bornminus_samples_list, born_samples_list, stein_params)
	

	# kappa_q_born_bornplus = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data,\
	# 												data_exact_dict, born_samples, born_samples_plus, score_approx, chi)
	# kappa_q_bornplus_born = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data, \
	# 												data_exact_dict, born_samples_plus, born_samples, score_approx, chi)
	# kappa_q_born_bornminus = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data,\
	# 												data_exact_dict, born_samples_list, born_samples_minus, score_approx, chi)
	# kappa_q_bornminus_born = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data,\
	# 												data_exact_dict, born_samples_minus, born_samples, score_approx, chi)
	
	L_stein_grad_1 = (1/(N_born_samples*N_bornplus_samples))*sum(kappa_q_born_bornplus)
	L_stein_grad_2 = (1/(N_born_samples*N_bornplus_samples))*sum(kappa_q_bornplus_born)
	L_stein_grad_3 = (1/(N_born_samples*N_bornminus_samples))*sum(kappa_q_born_bornminus)
	L_stein_grad_4 = (1/(N_born_samples*N_bornminus_samples))*sum(kappa_q_bornminus_born)

	L_stein_grad =  L_stein_grad_1 + L_stein_grad_2 + L_stein_grad_3 + L_stein_grad_4


	return L_stein_grad



def SteinCost(device_params, data_samples, data_exact_dict, born_samples, born_probs_dict,	
			N_kernel_samples, kernel_choice,
			stein_params):
	'''This function computes the Stein Discrepancy cost function between P and Q from samples from P and Q'''
	# sigma = [0.1, 10, 100]
	# kernel_array = GaussianKernelArray(samples, samples, sigma)
	
	device_name = device_params[0]
	as_qvm_value = device_params[1]
	qc = get_qc(device_name, as_qvm = as_qvm_value)
	qubits = qc.qubits()
	N_qubits = len(qubits)

	data_samples_list= SampleArrayToList(data_samples)
	born_samples_list= SampleArrayToList(born_samples)
	
	N_born_samples = len(born_samples_list)
	#Compute the empirical data distibution given samples

	kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_kernel_samples, kernel_choice)
	data_samples_list = SampleArrayToList(data_samples)

	kappa_q = ss.ComputeWeightedKernel(device_params, kernel_sampled_dict, data_samples_list, data_exact_dict, born_samples_list, born_samples_list, stein_params,'same')

	L_stein = (1/((N_born_samples)*(N_born_samples - 1)))*sum(kappa_q)

	return L_stein


