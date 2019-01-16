import math
import numpy as np
from auxiliary_functions import IntegerToString


"""Calculate the Hamming distance between two bit strings"""
def HammingWeight(s1, s2):
	assert len(s1) == len(s2)
	return sum(c1 != c2 for c1,c2 in zip(s1, s2))

"""Generate all bit strings between 0 and N_v and N_v to N_h"""
def Perms(n_v, n_h, m_h):
	s_visible = np.zeros((2**n_v, n_v))
	s_hidden = np.zeros((2**n_h, n_h))
	s_modes = np.zeros((2**m_h, m_h))

	#Visible units all permutations
	for v_index in range(0,2**n_v):
		v_string = IntegerToString(v_index, n_v)
		for v in range(0,n_v):
			s_visible[v_index, v] = float(v_string[v])

	#Hidden units all permutations
	for h_index in range(0,2**n_h):
		h_string = IntegerToString(h_index, n_h)
		for h in range(0, n_h):
			s_hidden[h_index][h] = float(h_string[h])

	#Hidden Modes all permutations, only used to compute training data - doesn't need to be outputted
	for hidden_mode_index in range(0,2**m_h):
		hidden_mode_string = IntegerToString(hidden_mode_index, m_h)
		for h in range(0,m_h):
			s_modes[hidden_mode_index][h] = float(hidden_mode_string[h])
	return s_visible, s_hidden, s_modes

"""Generates Random centre modes """
def CentreModes(n_v, m_h):
	s_cent = np.zeros((m_h, n_v))
	for h in range(0, m_h):
		#Fix random seed for reproducibility, for each centre mode, h
		np.random.seed(h)
		stemp = np.random.binomial(1, 0.5, n_v)
		#print(stemp)
		for v in range(0,n_v):
			s_cent[h][v] = stemp[v]

	return s_cent

"""Finds the Hamming weight of each possible input relative to each of the centre points"""
def HamWeightModes(s_visible, s_cent, n_v, m_h):
	hamweight = np.zeros((2**n_v, m_h))

	for string in range(0,2**n_v):
		for h in range(0,m_h):
			hamweight[string][h] = HammingWeight(s_cent[h][:], s_visible[:][string])

	return hamweight

"""This defines the full probability distribution over the visible nodes, and the hidden 'modes' """
def ProbDist(n_v, m_h, p, hamw):
	dist = np.zeros((2**n_v, m_h))
	for v_string in range(0,2**n_v):
		for h in range(0, m_h):
			dist[v_string][h] = ((p**(n_v - hamw[v_string][h]))*((1-p)**(hamw[v_string][h])))
	return dist

def all_binary_values(power):

    binary_list = np.zeros((2**power, power))

    for i in range(2**power):

        temp = i

        for j in range(power):

                binary_list[i,power - j - 1] = temp % 2
                temp >>= 1

    return binary_list

def TrainingData(N_v, N_h, M_h):
    """This function constructs example training data"""
    
    '''s_hidden/s_visible is all possible output strings of the qubits'''
    '''s_modes is all possible output strings over the modes'''
    
    # centre_modes = CentreModes(N_v, M_h)
    centre_modes = np.random.binomial(1, 0.5, (M_h,N_v))
    # bin_visible, _,_ = Perms(N_v, N_h, M_h)
    # print(bin_visible)
    bin_visible = all_binary_values(N_v)
    # print(bin_visible)
    hamweight = HamWeightModes(bin_visible, centre_modes, N_v, M_h)
    jointdist = ProbDist(N_v, M_h, 0.9, hamweight)
    data_dist = (1/M_h)*jointdist.sum(axis=1)
    #put data in dictionary
    data_dist_dict = {}
    for v_string in range(0,2**N_v):
    	bin_string_visible = IntegerToString(v_string, N_v)
    	data_dist_dict[bin_string_visible] = data_dist[v_string]
    return data_dist, data_dist_dict

# def DataSampler(N_v, N_h, M_h, N_samples, data_probs, exact_data_dict):
def DataSampler(N_v, N_h, M_h, N_samples, data_probs):
	'''This functions generates (N_samples) samples according to the given probability distribution
		data_probs'''
	
	'''Uncomment next line if a new run is required, i.e. not printing to file'''
	#data_dist, bin_visible, bin_hidden, data_dist_dict = TrainingData(N_v, N_h, M_h)

	#labels for output possibilities, integer in range [0, 2**N_v], corresponds to bitstring over N_v bits
	elements = []
	for i in range(0, 2**N_v):
		elements.append((IntegerToString(i, N_v)))

	data_samples = np.random.choice(elements, N_samples, True, data_probs)

	return data_samples

