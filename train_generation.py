import numpy as np

def ConvertToString(index, N_qubits):
	return "0" * (N_qubits-len(format(index,'b'))) + format(index,'b')

def TrainingData(N_v, N_h, M_h):
	"""This function constructs example training data"""

	'''s_hidden/s_visible is all possible output strings of the qubits'''
	'''s_modes is all possible output strings over the modes'''
	s_cent = np.zeros((M_h, N_v))
	s_visible = np.zeros((2**N_v, N_v))
	s_hidden = np.zeros((2**N_h, N_h))
	s_modes = np.zeros((2**M_h, M_h))
	hamweight = np.zeros((2**N_v, M_h))
	dist = np.zeros((2**N_v, M_h))

	"""Calculate the Hamming distance between two bit strings"""
	def HammingWeight(s1, s2):
		assert len(s1) == len(s2)
		return sum(c1 != c2 for c1,c2 in zip(s1, s2))

	"""Generate all bit strings between 0 and N_v and N_v to N_h"""
	def Perms(n_v, n_h, m_h):
		#Visible units all permutations
		for v_index in range(0,2**n_v):
			v_string = ConvertToString(v_index, N_h)
			for v in range(0,n_v):
				s_visible[v_string][v] = float(v_string[v])
		#Hidden units all permutations
		for h_index in range(0,2**n_h):
			h_string = ConvertToString(h_index, n_h)
			for h in range(0,n_h):
				s_hidden[h_index][h] = float(h_string[h])

		#Hidden Modes all permutations, only used to compute training data - doesn't need to be outputted
		for hidden_mode_index in range(0,2**m_h):
			hidden_mode_string = ConvertToString(hidden_mode_index, m_h)
			for h in range(0,m_h):
				s_modes[hidden_mode_index][h] = float(hidden_mode_string[h])
		return s_visible, s_hidden, s_modes

	"""Generates Random centre modes """
	def CentreModes(n_v, m_h):
		for h in range(0, m_h):
			stemp = np.random.binomial(1, 0.5, n_v)
			for v in range(0,n_v):
				s_cent[h][v] = stemp[v]

		return s_cent

	"""Finds the Hamming weight of each possible input relative to each of the centre points"""
	def HamWeightModes(s_visible, s_cent, n_v, m_h):
		for h in range(0, m_h):
			for string in range(0,2**n_v):
				hamweight[string][h] = HammingWeight(s_cent[h][:], s_visible[:][string])

		return hamweight

	"""This defines the full probability distribution over the visible nodes, and the hidden 'modes' """
	def ProbDist(n_v, m_h, p, hamw):
		dist_dict = {}
		for v_string in range(0,2**n_v):
			for h in range(0,m_h):
				dist[v_string][h] = ((p**(n_v - hamw[v_string][h]))*((1-p)**(hamw[v_string][h])))

		return dist

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


def DataSampler(N_v, N_h, M_h, N_samples):
	'''This functions generates (N_samples) samples according to the given probability distribution'''

	data_dist, bin_visible, bin_hidden = TrainingData(N_v, N_h, M_h)
	#labels for output possibilities, integer in range [0, 2**N_v], corresponds to bitstring over N_v bits
	elements = np.array([i for i in range(0, 2**N_v)])
	sample_string_array = np.zeros((N_samples, N_v))
	sample_integers = np.zeros((N_samples), dtype = np.int)

	for i in range(0, N_samples):
		sample_integers[i] = int(np.random.choice(elements, 1 , True, data_dist))
		s_temp = format(int(sample_integers[i]),'b')
		s_temp = "0" * (N_v-len(s_temp)) + s_temp
		#Assign each binary number in each string to array
		for v in range(0, N_v):
			sample_string_array[i][v] = float(s_temp[v])

	return sample_string_array, data_dist_dict
