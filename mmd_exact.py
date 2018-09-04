from sample_gen import BornSampler, PlusMinusSampleGen
from train_generation import TrainingData, DataSampler, ConvertToString
from classical_kernel import GaussianKernel, GaussianKernelExact
from file_operations_in import KernelDictFromFile
from mmd_kernel import KernelCircuit, KernelComputation, EncodingFunc


def SumContribution(N_qubits, dict_main, dict_one, dict_two, kernel_exact_dict):
	'''
	Computes the contribution to the MMD/MMD gradient for two rounds, first dict_main probabilities with
	dict_one probabilities, secondly dict_main with dict_two probabilities
	'''
	N_probs_main = len(dict_main)
	N_probs_one = len(dict_one)
	N_probs_two = len(dict_two)

	main_first = np.zeros((N_probs_main, N_probs_one))
	main_second = np.zeros((N_probs_main, N_probs_two))

	for main_term in range(0, N_probs_main):
		main_string = ConvertToString(main_term, N_qubits)
		for first_term in range(0, N_probs_one):
			first_string = ConvertToString(first_term, N_qubits)
			main_first[main_term, first_term] = dict_main[main_string]\
											*dict_one[first_string]*kernel_exact_dict[(main_string, first_string)]
		for second_term in range(0, N_probs_two):
			second_string = ConvertToString(N_probs_two, N_qubits)
			main_second[main_term, second_term] = dict_main[main_string]\
												*dict_two[second_string]*kernel_exact_dict[(main_string, second_string)]
	return main_first, main_second

def MMDKernelExact(N_v, bin_visible, N_k_samples, k_choice):
	#If the input corresponding to the kernel choice is either the gaussian kernel or the quantum kernel
	if (k_choice == 'Gaussian'):
		sigma = np.array([0.25, 10, 1000])
		k, k_exact_dict = GaussianKernelExact(N_v, bin_visible, sigma)
		#Gaussian approx kernel is equal to exact kernel
		k_exact = k
		k_dict = k_exact_dict
	elif (k_choice ==  'Quantum'):
		#compute for all binary strings
		ZZ, Z = EncodingFunc(N_v, bin_visible)
		k, k_exact, k_dict, k_exact_dict = KernelComputation(N_v, 2**N_v , 2**N_v, N_k_samples, ZZ, Z, ZZ, Z)
	else:
		print("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")

	#compute the expectation values for including each binary string sample
	return k, k_exact, k_dict, k_exact_dict

def MMDGradExact(N_v,
				data_exact_dict, born_probs_dict, born_probs_plus_dict, born_probs_minus_dict,
				N_k_samples, k_choice):
		#We have dictionaries of 5 files, the kernel, the data, the born probs, the born probs plus/minus
	k_exact_dict = KernelDictFromFile(N_v, N_k_samples, k_choice)
	#compute the term kp_1p_2 to add to expectation value for each pair, from bornplus/bornminus, with born machine and data
	born_plus, born_minus = SumContribution(N_v, born_probs_dict, born_probs_plus_dict, born_probs_minus_dict, k_exact_dict)
	data_plus, data_minus = SumContribution(N_v, data_exact_dict, born_probs_plus_dict, born_probs_minus_dict, k_exact_dict)

	#Return the gradient of the loss function (L = MMD^2) for a given parameter
	return 2*(born_plus.sum()-born_minus.sum()-data_minus.sum()+data_plus.sum())

def MMDCostExact(N_v, data_exact_dict, born_probs_dict, kernel_exact_dict):

	#compute the term kp_1p_2 to add to expectation value for each pair, from born/bornplus/bornminus machine, and data
	born_born, born_data = SumContribution(N_v, born_probs_dict, born_probs_dict, data_exact_dict, k_exact_dict)

	N_data= len(data_exact_dict)
	data_data = np.zeros((N_data, N_data))
	for sampledata1 in range(0, N_data):
		sampledata1_string= ConvertToString(sampledata1, N_v)
		for sampledata2 in range(0, N_born_probs):
			sampledata2_string= ConvertToString(sampledata2, N_v)
			data_data[sampledata1, sampledata2] = born_probs_dict[sampledata1_string]\
								*born_probs_dict[sampledata2_string]*kernel_exact_dict[(sampledata1_string, sampledata2_string)]

	#return the loss function (L = MMD^2)
	return born_born.sum() - 2*born_data.sum() + data_data.sum()
