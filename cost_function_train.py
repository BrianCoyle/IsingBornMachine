from pyquil.quil import Program
import numpy as np
from pyquil.api import get_qc

from sample_gen import BornSampler, PlusMinusSampleGen
from classical_kernel import GaussianKernel, GaussianKernelExact
from file_operations_in import KernelDictFromFile
from quantum_kernel import KernelCircuit, QuantumKernelComputation, EncodingFunc
from mmd_functions import MMDGrad, MMDCost
from stein_functions import SteinGrad, SteinCost
from auxiliary_functions import ConvertToString, EmpiricalDist, TotalVariationCost, MiniBatchSplit

################################################################################################################
#Train Model Using Stein Discrepancy with either exact kernel and gradient or approximate one using samples
################################################################################################################
def TrainBorn(device_params, cost_func,initial_params,
			N_epochs,  N_samples,
			data_train_test, data_exact_dict,
			k_choice, learning_rate, approx, score_approx, flag):
    device_name = device_params[0]
    as_qvm_value = device_params[1]

    qc = get_qc(device_name, as_qvm = as_qvm_value)
    qubits = qc.qubits() 
    #Initialise a 3 dim array for the graph weights, 2 dim array for biases and gamma parameters
    N_qubits = len(qubits)

    J 		= np.zeros((int(qubits[0]) + len(qubits), int(qubits[0]) + len(qubits), N_epochs))
    b 		= np.zeros((int(qubits[0]) + len(qubits), N_epochs))
    gamma_x = np.zeros((int(qubits[0]) + len(qubits), N_epochs))
    gamma_y = np.zeros((int(qubits[0]) + len(qubits), N_epochs))
    # #Import initial parameter values
    J[:,:,0] = initial_params['J']
    b[:,0] = initial_params['b']
    gamma_x[:,0] = initial_params['gamma_x']
    gamma_y[:,0] = initial_params['gamma_y']

    circuit_params = {}
    circuit_params[('J', 0)] = initial_params['J']
    circuit_params[('b', 0)] = initial_params['b']
    circuit_params[('gamma_x', 0)] = initial_params['gamma_x']
    circuit_params[('gamma_y', 0)] = initial_params['gamma_y']

    batch_size = N_samples[2]
    #Initialise the gradient arrays, each element is one parameter
    weight_grad     = np.zeros((int(qubits[-1])+1, int(qubits[-1])+1))
    bias_grad       = np.zeros((int(qubits[-1])+1))
    gamma_x_grad    = np.zeros((int(qubits[-1])+1))

    loss = {('Stein', 'Train'): [], ('MMD', 'Train'): [], 'TV': [],\
            ('Stein', 'Test'): [], ('MMD', 'Test'): []}

    born_probs_list = []
    empirical_probs_list = []

    stein_kernel_choice = 'Quantum'
    chi = 0.01
    circuit_choice ='QAOA'
    for epoch in range(0, N_epochs-1):
        #gamma_x/gamma_y is not to be trained, set gamma values to be constant at each epoch
        gamma_x[:,epoch + 1] = gamma_x[:, epoch]
        gamma_y[:,epoch + 1] = gamma_y[:, epoch]
        # print('gamma for epoch', gamma_x[:, epoch])
        # print('bias for epoch', b[:, epoch])

        print("\nThis is Epoch number: ", epoch)
        # Jt = J[:,:,epoch]
        # bt = b[:,epoch]
        # gxt = gamma_x[:,epoch]
        # gyt = gamma_y[:,epoch]
        circuit_params_per_epoch = {}

        circuit_params_per_epoch['J'] = J[:,:,epoch]
        circuit_params_per_epoch['b'] = b[:,epoch]
        circuit_params_per_epoch['gamma_x'] = gamma_x[:,epoch]
        circuit_params_per_epoch['gamma_y'] = gamma_y[:,epoch]

        circuit_params[('J', epoch)] = J[:,:,epoch]
        circuit_params[('b', epoch)] = b[:,epoch]
        circuit_params[('gamma_x', epoch)] = gamma_x[:,epoch]
        circuit_params[('gamma_y', epoch)] = gamma_y[:,epoch]

        #generate samples, and exact probabilities for current set of parameters
        born_samples, born_probs_dict = BornSampler(device_params, N_samples, circuit_params_per_epoch, circuit_choice)
        
        born_probs_list.append(born_probs_dict)
        
        empirical_probs_dict = EmpiricalDist(born_samples, N_qubits)
        empirical_probs_list.append(empirical_probs_dict)

        print('The Born Machine Outputs Probabilites\n',born_probs_dict)
        print('The Data is\n,', data_exact_dict)

        # print('The Empirical Born is\n,', empirical_probs_dict)
        
        '''Updating bias b[r], control set to 'BIAS' '''
        for bias_index in qubits:
            born_samples_plus, born_samples_minus\
                                    = PlusMinusSampleGen(device_params, circuit_params_per_epoch,\
                                                0,0, bias_index, 0, \
                                                circuit_choice, 'BIAS',  N_samples)

            #Shuffle all samples to avoid bias in Minibatch Training
            np.random.shuffle(data_train_test[0])
            np.random.shuffle(born_samples)
            np.random.shuffle(born_samples_plus)
            np.random.shuffle(born_samples_minus)

            #Use only first 'batch_size' number of samples for each update
            if batch_size > len(data_train_test[0]) or batch_size > len(born_samples):
                raise IOError('The batch size is too large')
            else:
                data_batch = MiniBatchSplit(data_train_test[0], batch_size)
                born_batch = MiniBatchSplit(born_samples, batch_size)
                # bornplus_batch = MiniBatchSplit(born_samples_plus, batch_size)
                # bornminus_batch = MiniBatchSplit(born_samples_minus, batch_size)

            ##If the exact MMD is to be computed approx == 'Exact', if only approximate version using samples, approx == 'Sampler'
            #Trained using only data in training set
            if (cost_func == 'Stein'):
                bias_grad[bias_index] = SteinGrad(device_params, data_batch, data_exact_dict,\
                                        born_batch, born_probs_dict,\
                                        born_samples_plus, \
                                        born_samples_minus, \
                                        N_samples, k_choice, approx, score_approx, chi, stein_kernel_choice)
            elif (cost_func == 'MMD'):
                bias_grad[bias_index] = MMDGrad(device_params, data_batch, data_exact_dict,
                                        born_batch, born_probs_dict,\
                                        born_samples_plus,  \
                                        born_samples_minus, \
                                        N_samples, k_choice, approx, flag)
            else: raise IOError('\'cost_func\' must be either \'Stein\', or \'MMD\' ')
            # print('bias_grad:', bias_grad)

            b[:, epoch + 1] = b[:, epoch] - learning_rate*bias_grad

        # if (circuit_choice == 'QAOA'):	
        # 	'''Updating finalparam gamma[s], control set to 'FINALPARAM' '''
        # 	for gammaindex in range(0,N):
        #            born_samples_plus, born_samples_plus \
        # 							= PlusMinusSampleGen(device_params, N_h, \
        # 											circuit_params_per_epoch, \
        # 											0, 0, 0, gammaindex, \
        # 											circuit_choice, 'GAMMA',\
        # 											N_samples)
        # 		# Flip ordering of samples to be consistent with Rigetti convention
        # 		born_samples_plus = np.flip(born_samples_plus_unflip, 1)
        # 		born_samples_minus = np.flip(born_samples_minus_unflip, 1)
            
        # 		##If the exact MMD is to be computed approx == 'Exact', if only approximate version using samples, approx == 'Sampler'
        # 		gamma_x_grad[gammaindex] = SteinGrad(device_params, data_train_test[0], data_exact_dict,\
        # 								born_samples, born_probs_dict,\
        # 								born_samples_plus, born_plus_exact_dict, \
        # 								born_samples_minus, born_minus_exact_dict,\
        # 								N_samples, k_choice, approx, score_approx, chi, stein_kernel_choice)
            
        # 	# print('gamma_x_grad is:', gamma_x_grad)
        # 	gamma_x[:, epoch + 1] = gamma_x[:, epoch] - learning_rate*gamma_x_grad

        '''Updating weight J[p,q], control set to 'WEIGHTS' '''
        for q in qubits:
            for p in qubits:
                if (p < q):
                    ## Draw samples from +/- pi/2 shifted circuits for each weight update, J_{p, q}
                    born_samples_plus, born_samples_plus \
                                    = PlusMinusSampleGen(device_params, circuit_params_per_epoch, \
                                                        p, q , 0, 0,\
                                                        circuit_choice, 'WEIGHTS', N_samples)
            
                    #Shuffle all samples to avoid bias in Minibatch Training
                    np.random.shuffle(data_train_test[0])
                    np.random.shuffle(born_samples)
                    np.random.shuffle(born_samples_plus)
                    np.random.shuffle(born_samples_minus)

                    #Use only first 'batch_size' number of samples for each update
                    if (batch_size > len(data_train_test[0]) or batch_size > len(born_samples)):
                        raise IOError('The batch size is too large')
                    else:
                        data_batch      = MiniBatchSplit(data_train_test[0], batch_size)
                        born_batch      = MiniBatchSplit(born_samples, batch_size)


                    ##If the exact MMD is to be computed approx == 'Exact', if only approximate version using samples, approx == 'Sampler'
                    if (cost_func == 'Stein'):
                        weight_grad[p,q] = SteinGrad(device_params, data_batch, data_exact_dict,\
                                                    born_batch, born_probs_dict,\
                                                    born_samples_plus, \
                                                    born_samples_minus,\
                                                    N_samples, k_choice, approx, score_approx, chi, stein_kernel_choice)

                    elif (cost_func == 'MMD'):
                        weight_grad[p,q] = MMDGrad(device_params, data_batch, data_exact_dict,
                                            born_batch, born_probs_dict,\
                                            born_samples_plus,  \
                                            born_samples_minus,\
                                            N_samples, k_choice, approx, flag)
                    else: raise IOError('\'cost_func\' must be either \'Stein\', or \'MMD\' ')

        J[:,:, epoch+1] = J[:,:, epoch] - learning_rate*(weight_grad + np.transpose(weight_grad))

        # #Check Stein Discrepancy of Model Distribution with training set
        # loss[('Stein', 'Train')].append(SteinCost(device_params, data_train_test[0], data_exact_dict, born_samples,\
        #                             born_probs_dict, N_samples, k_choice, approx, score_approx, chi, stein_kernel_choice))

        # #Check Stein Discrepancy of Model Distribution with test set
        # loss[('Stein', 'Test')].append(SteinCost(device_params, data_train_test[1], data_exact_dict, born_samples,\
        #                             born_probs_dict, N_samples, k_choice, approx, score_approx, chi, stein_kernel_choice))

        #Check MMD of Model Distribution with training set: data_train_test[0]
        loss[('MMD', 'Train')].append(MMDCost(device_params, data_train_test[0], data_exact_dict, born_samples,born_probs_dict, N_samples, k_choice, approx, flag))


        #Check MMD of Model Distribution with test set: data_train_test[1]
        loss[('MMD', 'Test')].append(MMDCost(device_params, data_train_test[1], data_exact_dict, born_samples,born_probs_dict, N_samples, k_choice, approx, flag))

        # #Check Total Variation Distribution
        # loss[('TV')].append(TotalVariationCost(data_exact_dict, born_probs_dict))


        # print("The Stein Discrepancy for epoch ", epoch, "is", loss[('Stein', 'Train')][epoch])
        # print("The Variation Distance for epoch ", epoch, "is", loss['TV'][epoch])
        print("The MMD Loss for epoch ", epoch, "is", loss[('MMD', 'Train')][epoch])
        

    return loss, circuit_params, born_probs_list, empirical_probs_list



def KernelExact(device_params, bin_visible, N_samples, kernel_choice):
	#If the input corresponding to the kernel choice is either the gaussian kernel or the quantum kernel
    device_name = device_params[0]
    as_qvm_value = device_params[1]

    qc = get_qc(device_name, as_qvm = as_qvm_value)
    qubits = qc.qubits()
    N_qubits =len(qubits)
    if (kernel_choice == 'Gaussian'):
        sigma = np.array([0.25, 10, 1000])
        k, k_exact_dict = GaussianKernelExact(N_qubits, bin_visible, sigma)
        #Gaussian approx kernel is equal to exact kernel
        k_exact = k
        k_dict = k_exact_dict
    elif (kernel_choice ==  'Quantum'):
        #compute for all binary strings
        encoded_samples = EncodingFunc(N_qubits, bin_visible)
        k, k_exact, k_dict, k_exact_dict = QuantumKernelComputation(device_params, 2**N_qubits , 2**N_qubits, N_samples, encoded_samples, encoded_samples)
    else: raise IOError("Please enter either 'Gaussian' or 'Quantum' to choose a kernel")

    #compute the expectation values for including each binary string sample
    return k, k_exact, k_dict, k_exact_dict
