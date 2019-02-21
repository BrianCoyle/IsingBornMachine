import numpy as np
from auxiliary_functions import L2Norm,SampleListToArray
from file_operations_in import DataImport
import feydy_sinkhorn as feydy_sink
import torch
import auxiliary_functions as aux

'''
    This function computes the Sinkhorn Cost function, and its gradient, following the Method of:
    Interpolating between Optimal Transport and MMD using Sinkhorn Divergences by Feydy et. al.
    https://arxiv.org/abs/1810.08278
    
    The functions used from feydy_sinkhorn are those in https://github.com/jeanfeydy/global-divergences in common/sinkhorn_balanced_simple
    to directly compute the Sinkhorn Divergence, and adapted from there to compute its gradient. 
'''



def FeydySink(born_samples_all, data_samples_all, epsilon):
    '''
    Converts sample vectors to required format, first to the empirical distribution,
    then into torch tensors to feed into Feydy Implementation of the Sinkhorn Divergence
    '''
    #Extract samples (unrepeated) as array, along with corresponding empirical probabilities
    # Convert everything to pytorch tensors to be compatible with Feydy library
    _, _, born_samples_tens, born_probs_tens = aux.ExtractSampleInformation(born_samples_all)
    _, _, data_samples_tens, data_probs_tens = aux.ExtractSampleInformation(data_samples_all)

    sinkhorn_feydy = feydy_sink.sinkhorn_divergence(born_probs_tens, born_samples_tens,  data_probs_tens, data_samples_tens)

    return sinkhorn_feydy

def SinkGrad(born_samples, born_samples_pm, data_samples, epsilon):
    '''
    Converts sample vectors for gradient to required format, first to the empirical distribution,
    then into torch tensors to feed into Feydy Implementation of the Sinkhorn Divergence
    '''
    [born_samples_plus, born_samples_minus] = born_samples_pm

    #The following extracts the empirical probabilities, and corresponding sample values from the 
    #arrays of samples. These are then converted into pytorch tensors to be compatable with functions in feydy_sinkhorn
    _, _, born_samples_tens,       born_probs_tens          = aux.ExtractSampleInformation(born_samples)
    _, _, born_plus_samples_tens,  born_plus_probs_tens     = aux.ExtractSampleInformation(born_samples_plus)
    _, _, born_minus_samples_tens, born_minus_probs_tens    = aux.ExtractSampleInformation(born_samples_minus)
    _, _, data_samples_tens,       data_probs_tens          = aux.ExtractSampleInformation(data_samples)


    g_data  , f_born = feydy_sink.sink(born_probs_tens, born_samples_tens,  data_probs_tens, data_samples_tens)# Compute optimal dual vectors between born samples and data
    _       , s_born = feydy_sink.sym_sink(born_probs_tens, born_samples_tens)  #Compute autocorrelation vectors for Born data

 
    p=2 #Sinkhorn Cost will be Euclidean Distance squared, or l_2 norm squared. Equivalent to Hamming Distance in this case.

    cost_matrix_plus_born = feydy_sink.dist_matrix(born_plus_samples_tens,   born_samples_tens, p, epsilon) #Compute cost matrix between born_plus and born samples
    cost_matrix_minus_born = feydy_sink.dist_matrix(born_minus_samples_tens, born_samples_tens, p, epsilon) #Compute cost matrix between born_minus and born samples
    
    cost_matrix_plus_data = feydy_sink.dist_matrix(born_plus_samples_tens,      data_samples_tens,  p,    epsilon) #Compute cost matrix between born_plus and data samples
    cost_matrix_minus_data = feydy_sink.dist_matrix(born_minus_samples_tens,    data_samples_tens,  p,    epsilon) #Compute cost matrix between born_minus and data samples
  
    f_plus = -epsilon*feydy_sink.lse((g_data + data_probs_tens.log().view(1, -1)).view(1, -1)       - cost_matrix_plus_data )
    s_plus = -epsilon*feydy_sink.lse((s_born  + born_probs_tens.log().view(1, -1)).view(1, -1)     - cost_matrix_plus_born )

    f_minus = -epsilon*feydy_sink.lse((g_data  + data_probs_tens.log().view(1, -1)).view(1, -1)     - cost_matrix_minus_data)
    s_minus = -epsilon*feydy_sink.lse((s_born + born_probs_tens.log().view(1, -1)).view(1, -1)      - cost_matrix_minus_born)
   
    feydy_sink_grad =feydy_sink.scal(born_minus_probs_tens, f_minus - s_minus) -  feydy_sink.scal(born_plus_probs_tens, f_plus - s_plus)

    return feydy_sink_grad








