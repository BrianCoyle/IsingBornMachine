
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
				N_samples, k_choice, approx, score_approx, chi, stein_kernel_choice):

	device_name = device_params[0]
	as_qvm_value = device_params[1]

	qc = get_qc(device_name, as_qvm = as_qvm_value)

	qubits = qc.qubits()
	N_qubits = len(qubits)
	
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
	kappa_q_born_bornplus = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data_samples_list,\
													data_exact_dict, born_samples_list, bornplus_samples_list, score_approx, chi)
	kappa_q_bornplus_born = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data_samples_list, \
													data_exact_dict, bornplus_samples_list,born_samples_list, score_approx, chi)
	kappa_q_born_bornminus = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data_samples_list,\
													data_exact_dict, born_samples_list, bornminus_samples_list, score_approx, chi)
	kappa_q_bornminus_born = ss.ComputeWeightedKernel(device_params, kernel_dict_for_stein, data_samples_list,\
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


	L_stein_grad =  sum(list(expectation_value_summand_1.values()))- sum(list(expectation_value_summand_2.values())) +  \
					sum(list(expectation_value_summand_3.values()))- sum(list(expectation_value_summand_4.values()))


	return L_stein_grad



def SteinCost(device_params, data_samples, data_exact_dict, born_samples, born_probs_dict,	
			N_kernel_samples, kernel_choice,
			approx, score_approx, chi, stein_kernel_choice):
	'''This function computes the Stein Discrepancy cost function between P and Q from samples from P and Q'''

	device_name = device_params[0]
	as_qvm_value = device_params[1]

	qc = get_qc(device_name, as_qvm = as_qvm_value)

	qubits = qc.qubits()
	N_qubits = len(qubits)
	data_samples_list= SampleArrayToList(data_samples)
	born_samples_list= SampleArrayToList(born_samples)
	
	#Compute the empirical data distibution given samples

	kernel_sampled_dict  = KernelDictFromFile(N_qubits, N_kernel_samples, kernel_choice)
	data_samples_list = SampleArrayToList(data_samples)
	emp_born_dist = EmpiricalDist(born_samples, N_qubits)

	# expectation_value_summand = np.zeros((len(born_samples)))
	expectation_value_summand = {}

	kappa_q = ss.ComputeWeightedKernel(device_params, kernel_sampled_dict, data_samples_list, data_exact_dict, born_samples_list, born_samples_list, score_approx, chi)
	
	for sample1 in born_samples_list:
		for sample2 in born_samples_list:
			if (sample1, sample2) in kappa_q:
					expectation_value_summand[(sample1, sample2)] = emp_born_dist[sample1]*kappa_q[(sample1, sample2)]*emp_born_dist[sample2]

	L_stein =   sum(list(expectation_value_summand.values()))


	return L_stein



