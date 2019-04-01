from pyquil.quil import Program
import numpy as np

from sample_gen import BornSampler, PlusMinusSampleGen
from cost_functions import CostFunction, CostGrad
from auxiliary_functions import  EmpiricalDist, TotalVariationCost, MiniBatchSplit

################################################################################################################
#Train Model Using Stein Discrepancy with either exact kernel and gradient or approximate one using samples
################################################################################################################
def TrainBorn(qc, cost_func,initial_params,
                N_epochs,  N_samples,
                data_train_test, data_exact_dict,
                k_choice, flag, learning_rate_init, 
                stein_params, 
                sinkhorn_eps):
            
    N_qubits = len(qc.qubits())
    #Import initial parameter values
    circuit_params = {}
    circuit_params[('J', 0)] = initial_params['J']
    circuit_params[('b', 0)] = initial_params['b']
    circuit_params[('gamma', 0)] = initial_params['gamma']
    circuit_params[('delta', 0)] = initial_params['delta']
    circuit_params[('sigma', 0)] = initial_params['sigma']

    batch_size = N_samples[2]
    #Initialise the gradient arrays, each element is one parameter
    weight_grad     = np.zeros((N_qubits, N_qubits))
    bias_grad       = np.zeros((N_qubits))

    loss = {('MMD', 'Train'): [], ('MMD', 'Test'): [],\
            ('Stein', 'Train'): [], ('Stein', 'Test'): [], \
            ('Sinkhorn', 'Train'): [], ('Sinkhorn', 'Test'): [], \
            'TV': []}

    born_probs_list = []
    empirical_probs_list = []

    circuit_choice ='QAOA'
    
    #Initialize momentum vectors at 0 for Adam optimiser
    [m_bias, v_bias] = [np.zeros((N_qubits)) for _ in range(2)] 
    [m_weights, v_weights] = [np.zeros((N_qubits, N_qubits)) for _ in range(2)] 

    for epoch in range(0, N_epochs-1):

        #gamma/delta is not to be trained, set gamma values to be constant at each epoch
        circuit_params[('gamma', epoch+1)] = circuit_params[('gamma', epoch)]
        circuit_params[('delta', epoch+1)] = circuit_params[('delta', epoch)]

        print("\nThis is Epoch number: ", epoch)
     
        circuit_params_per_epoch = {}

        circuit_params_per_epoch['J'] = circuit_params[('J', epoch)]
        circuit_params_per_epoch['b'] = circuit_params[('b', epoch)]
        circuit_params_per_epoch['gamma'] =  circuit_params[('gamma', epoch)]
        circuit_params_per_epoch['delta'] = circuit_params[('delta', epoch)]

        #generate samples, and exact probabilities for current set of parameters
        born_samples, born_probs_approx_dict, born_probs_exact_dict = BornSampler(qc, N_samples, circuit_params_per_epoch, circuit_choice)
        # print(born_samples)
                
        born_probs_list.append(born_probs_approx_dict)
        empirical_probs_list.append(born_probs_approx_dict)
        # print('The Empirical Data is:', EmpiricalDist(data_train_test[0], N_qubits ))
        print('The Born Machine Outputs Probabilites\n', born_probs_approx_dict)
        print('The Data is\n,', data_exact_dict)

        loss[(cost_func, 'Train')].append(CostFunction(qc, cost_func, data_train_test[0], data_exact_dict, born_samples,\
                                                        born_probs_approx_dict, N_samples, k_choice, stein_params, flag, sinkhorn_eps))
        loss[(cost_func, 'Test')].append(CostFunction(qc, cost_func, data_train_test[1], data_exact_dict, born_samples,\
                                                        born_probs_approx_dict, N_samples, k_choice, stein_params, flag, sinkhorn_eps))

        print("The %s Loss for epoch " %cost_func, epoch, "is", loss[(cost_func, 'Train')][epoch])
        
        #Check Total Variation Distribution using the exact output probabilities
        loss[('TV')].append(TotalVariationCost(data_exact_dict, born_probs_exact_dict))

        print("The Variation Distance for epoch ", epoch, "is", loss['TV'][epoch])
        '''Updating bias b[r], control set to 'BIAS' '''
        for bias_index in range(0, N_qubits):
            born_samples_pm = PlusMinusSampleGen(qc, circuit_params_per_epoch,\
                                                0,0, bias_index, 0, \
                                                circuit_choice, 'BIAS',  N_samples)

            #Shuffle all samples to avoid bias in Minibatch Training
            np.random.shuffle(data_train_test[0])
            np.random.shuffle(born_samples)

            #Use only first 'batch_size' number of samples for each update
            if batch_size > len(data_train_test[0]) or batch_size > len(born_samples):
                raise IOError('The batch size is too large')
            else:
                data_batch = MiniBatchSplit(data_train_test[0], batch_size)
                born_batch = MiniBatchSplit(born_samples, batch_size)

    
            bias_grad[bias_index] = CostGrad(qc, cost_func, data_batch, data_exact_dict,
                                                born_batch, born_probs_approx_dict,
                                                born_samples_pm, 
                                                N_samples, k_choice, stein_params, flag, sinkhorn_eps)
            #Update biases for next epoch

        '''Updating weight J[p,q], control set to 'WEIGHTS' '''
        for q in range(0, N_qubits):
            for p in range(0, N_qubits):
                if (p < q):
                    ## Draw samples from +/- pi/2 shifted circuits for each weight update, J_{p, q}
                    born_samples_pm = PlusMinusSampleGen(qc, circuit_params_per_epoch, \
                                                        p, q, 0, 0,\
                                                        circuit_choice, 'WEIGHTS', N_samples)
            
                    #Shuffle all samples to avoid bias in Minibatch Training
                    np.random.shuffle(data_train_test[0])
                    np.random.shuffle(born_samples)

                    #Use only first 'batch_size' number of samples for each update
                    if (batch_size > len(data_train_test[0]) or batch_size > len(born_samples)):
                        raise IOError('The batch size is too large')
                    else:
                        data_batch      = MiniBatchSplit(data_train_test[0], batch_size)
                        born_batch      = MiniBatchSplit(born_samples, batch_size)


                    weight_grad[p,q] = CostGrad(qc, cost_func, data_batch, data_exact_dict,
                                                born_batch, born_probs_approx_dict,
                                                born_samples_pm, 
                                                N_samples, k_choice, stein_params, flag, sinkhorn_eps)
        
        #Update Weights for next epoch
        learning_rate_bias, m_bias, v_bias          = AdamLR(learning_rate_init, epoch, bias_grad, m_bias, v_bias)
        learning_rate_weights, m_weights, v_weights = AdamLR(learning_rate_init, epoch, weight_grad + np.transpose(weight_grad), m_weights, v_weights)

        circuit_params[('b', epoch+1)] = circuit_params[('b', epoch)] - learning_rate_bias*bias_grad
        circuit_params[('J', epoch+1)] = circuit_params[('J', epoch)] - learning_rate_weights*(weight_grad + np.transpose(weight_grad))

    
    return loss, circuit_params, born_probs_list, empirical_probs_list


def AdamLR(learning_rate_init, timestep, gradient, m, v, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    '''
    Method to compute Adam learning rate which includes momentum
    Parameters, beta1, beta2, epsilon are as recommended in orginal Adam paper
    '''
    timestep = timestep +1
    m           = np.multiply(beta1, m) + np.multiply((1-beta1), gradient)
    v           = np.multiply(beta2, v) + np.multiply((1-beta2) , gradient**2)
    corrected_m = np.divide(m , (1- beta1**timestep))
    corrected_v = np.divide(v, (1- beta2**timestep))

    return learning_rate_init*(np.divide(corrected_m, np.sqrt(corrected_v)+ epsilon)), m, v
