import numpy as np
from collections import Counter

def ConvertToString(index, N_qubits):
	return "0" * (N_qubits-len(format(index,'b'))) + format(index,'b')

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
		v_string = ConvertToString(v_index, n_v)
		for v in range(0,n_v):
			s_visible[v_index, v] = float(v_string[v])

	#Hidden units all permutations
	for h_index in range(0,2**n_h):
		h_string = ConvertToString(h_index, n_h)
		for h in range(0, n_h):
			s_hidden[h_index][h] = float(h_string[h])

	#Hidden Modes all permutations, only used to compute training data - doesn't need to be outputted
	for hidden_mode_index in range(0,2**m_h):
		hidden_mode_string = ConvertToString(hidden_mode_index, m_h)
		for h in range(0,m_h):
			s_modes[hidden_mode_index][h] = float(hidden_mode_string[h])
	return s_visible, s_hidden, s_modes

"""Generates Random centre modes """
def CentreModes(n_v, m_h):
	s_cent = np.zeros((m_h, n_v))
	for h in range(0, m_h):
		stemp = np.random.binomial(1, 0.5, n_v)
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
		for h in range(0,m_h):
			dist[v_string][h] = ((p**(n_v - hamw[v_string][h]))*((1-p)**(hamw[v_string][h])))
	return dist

def TrainingData(N_v, N_h, M_h):
	"""This function constructs example training data"""

	'''s_hidden/s_visible is all possible output strings of the qubits'''
	'''s_modes is all possible output strings over the modes'''

	centre_modes = CentreModes(N_v, M_h)
	bin_visible, bin_hidden, bin_modes = Perms(N_v, N_h, M_h)
	hamweight = HamWeightModes(bin_visible, centre_modes, N_v, M_h)
	jointdist = ProbDist(N_v, M_h, 0.9, hamweight)
	data_dist = (1/M_h)*jointdist.sum(axis=1)
	#put data in dictionary
	data_dist_dict = {}
	for v_string in range(0,2**N_v):
		bin_string_visible =ConvertToString(v_string, N_v)
		data_dist_dict[bin_string_visible] = data_dist[v_string]
	return data_dist, bin_visible, bin_hidden, data_dist_dict


def EmpiricalDist(samples, N_v, *arg):
	'''This method outputs the empirical probability distribution given samples in a numpy array
		as a dictionary, with keys as outcomes, and values as probabilities'''
		
	if type(samples) is not np.ndarray and type(samples) is not list:
		raise TypeError('The samples must be either a numpy array, or list')

	if type(samples) is np.ndarray:
		N_samples = samples.shape[0]
		string_list = []
		for sample in range(0, N_samples):
			'''Convert numpy array of samples, to a list of strings of the samples to put in dict'''
			string_list.append(''.join(map(str, samples[sample, :].tolist())))

	elif type(samples) is list:
		N_samples = len(samples)
		string_list = samples

	counts = Counter(string_list)

	for element in counts:
		'''Convert occurances to relative frequencies of binary string'''
		counts[element] = counts[element]/(N_samples)

	for index in range(0, 2**N_v):
		'''If a binary string has not been seen in samples, set its value to zero'''
		if ConvertToString(index, N_v) not in counts:
			counts[ConvertToString(index, N_v)] = 0

	sorted_samples_dict = {}

	keylist = sorted(counts)
	for key in keylist:
		sorted_samples_dict[key] = counts[key]

	return sorted_samples_dict

def DataSampler(N_v, N_h, M_h, N_samples, data_probs, exact_data_dict):
	'''This functions generates (N_samples) samples according to the given probability distribution
		data_probs'''
	
	'''Uncomment next line if a new run is required, i.e. not printing to file'''
	#data_dist, bin_visible, bin_hidden, data_dist_dict = TrainingData(N_v, N_h, M_h)

	#labels for output possibilities, integer in range [0, 2**N_v], corresponds to bitstring over N_v bits
	elements = []
	for i in range(0, 2**N_v):
		elements.append((ConvertToString(i, N_v)))

	data_samples = np.random.choice(elements, N_samples, True, data_probs)

	#compute empirical distirbution of the samples
	dist_dict = EmpiricalDist(data_samples.tolist(), N_v)

	return data_samples

