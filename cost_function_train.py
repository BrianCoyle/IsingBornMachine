from pyquil.quil import Program
import numpy as np
from pyquil.api import get_qc

from sample_gen import BornSampler, PlusMinusSampleGen
from mmd_functions import MMDGrad, MMDCost
from stein_functions import SteinGrad, SteinCost
from auxiliary_functions import ConvertToString, EmpiricalDist, TotalVariationCost, MiniBatchSplit

################################################################################################################
#Train Model Using Stein Discrepancy with either exact kernel and gradient or approximate one using samples
################################################################################################################
def TrainBorn(device_params, cost_func,initial_params,
                N_epochs,  N_samples,
                data_train_test, data_exact_dict,
                k_choice, flag):
            
    device_name = device_params[0]
    as_qvm_value = device_params[1]
    qc = get_qc(device_name, as_qvm = as_qvm_value)
    qubits = qc.qubits() 
    #Initialise a 3 dim array for the graph weights, 2 dim array for biases and gamma parameters
    N_qubits = len(qubits)

    #Import initial parameter values
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

    stein_params = {}

    if cost_func == 'Stein':
        stein_params[0] = 'Exact_Score'     #Choice of method to approximate Stein Score:                   stein_score
        stein_params[1] = 6                 #Number of Nystrom Eigenvectors, J for spectral_stein method:   J
        stein_params[2] = 0.01              #regularization paramter for identity_stein method:             \chi
        stein_params[3] = 'Gaussian'        #Kernel for computing Stein Score:                              stein_kernel_choice
        stein_params[4] = [0.1, 10, 100]    #Bandwidth parameters for Gaussian Kernel:                      sigma

    circuit_choice ='QAOA'

    for epoch in range(0, N_epochs-1):
        learning_rate = 0.1*(1/(epoch+1))
        #gamma_x/gamma_y is not to be trained, set gamma values to be constant at each epoch
        circuit_params[('gamma_x', epoch+1)] = circuit_params[('gamma_x', epoch)]
        circuit_params[('gamma_y', epoch+1)] = circuit_params[('gamma_y', epoch)]

        print("\nThis is Epoch number: ", epoch)
     
        circuit_params_per_epoch = {}

        circuit_params_per_epoch['J'] = circuit_params[('J', epoch)]
        circuit_params_per_epoch['b'] = circuit_params[('b', epoch)]
        circuit_params_per_epoch['gamma_x'] =  circuit_params[('gamma_x', epoch)]
        circuit_params_per_epoch['gamma_y'] = circuit_params[('gamma_y', epoch)]

        #generate samples, and exact probabilities for current set of parameters
        born_samples, born_probs_approx_dict, born_probs_exact_dict = BornSampler(device_params, N_samples, circuit_params_per_epoch, circuit_choice)
                
                
        born_probs_list.append(born_probs_approx_dict)
        empirical_probs_list.append(born_probs_approx_dict)

        print('The Born Machine Outputs Probabilites\n', born_probs_approx_dict)
        print('The Data is\n,', data_exact_dict)

        
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
                                        born_batch, born_probs_approx_dict,\
                                        born_samples_plus, \
                                        born_samples_minus, \
                                        N_samples, k_choice, stein_params)
            elif (cost_func == 'MMD'):
                bias_grad[bias_index] = MMDGrad(device_params, data_batch, data_exact_dict,
                                        born_batch, born_probs_approx_dict,\
                                        born_samples_plus,  \
                                        born_samples_minus, \
                                        N_samples, k_choice, flag)
            else: raise IOError('\'cost_func\' must be either \'Stein\', or \'MMD\' ')

            circuit_params[('b', epoch+1)] = circuit_params[('b', epoch)]\
                                                - learning_rate*bias_grad

        # if (circuit_choice == 'QAOA'):	
        # 	'''Updating finalparam gamma[s], control set to 'FINALPARAM' '''
        # 	for gammaindex in range(0,N):
        #            born_samples_plus, born_samples_plus \
        # 							= PlusMinusSampleGen(device_params, N_h, \
        # 											circuit_params_per_epoch, \
        # 											0, 0, 0, gammaindex, \
        # 											circuit_choice, 'GAMMA',\
        # 											N_samples)
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
                                                    born_batch, born_probs_approx_dict,\
                                                    born_samples_plus, \
                                                    born_samples_minus,\
                                                    N_samples, k_choice, stein_params)

                    elif (cost_func == 'MMD'):
                        weight_grad[p,q] = MMDGrad(device_params, data_batch, data_exact_dict,
                                            born_batch, born_probs_approx_dict,\
                                            born_samples_plus,  \
                                            born_samples_minus,\
                                            N_samples, k_choice,flag)
                    else: raise IOError('\'cost_func\' must be either \'Stein\', or \'MMD\' ')

        circuit_params[('J', epoch+1)] = circuit_params[('J', epoch)]  \
                                        - learning_rate*(weight_grad + np.transpose(weight_grad))

        if cost_func == 'Stein':
            #Check Stein Discrepancy of Model Distribution with training set
            loss[('Stein', 'Train')].append(SteinCost(device_params, data_train_test[0], data_exact_dict, born_samples,\
                                        born_probs_approx_dict, N_samples, k_choice, stein_params))

            #Check Stein Discrepancy of Model Distribution with test set
            loss[('Stein', 'Test')].append(SteinCost(device_params, data_train_test[1], data_exact_dict, born_samples,\
                                        born_probs_approx_dict, N_samples, k_choice, stein_params))
            
            print("The Stein Discrepancy for epoch ", epoch, "is", loss[('Stein', 'Train')][epoch])

        elif cost_func == 'MMD':
            #Check MMD of Model Distribution with training set: data_train_test[0]
            loss[('MMD', 'Train')].append(MMDCost(device_params, data_train_test[0], data_exact_dict, born_samples, born_probs_approx_dict, N_samples, k_choice, flag))


            #Check MMD of Model Distribution with test set: data_train_test[1]
            loss[('MMD', 'Test')].append(MMDCost(device_params, data_train_test[1], data_exact_dict, born_samples, born_probs_approx_dict, N_samples, k_choice, flag))
            
            print("The MMD Loss for epoch ", epoch, "is", loss[('MMD', 'Train')][epoch])

        elif cost_func == 'TV':
            #Check Total Variation Distribution using the exact output probabilities
            loss[('TV')].append(TotalVariationCost(data_exact_dict, born_probs_exact_dict))

            print("The Variation Distance for epoch ", epoch, "is", loss['TV'][epoch])

    return loss, circuit_params, born_probs_list, empirical_probs_list