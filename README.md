# IsingBornMachine
Implementation of Quantum Ising Born Machine using Rigetti Forest Platform, with two approaches: 
1. Computing the KL Divergence, 
2. MMD as a cost function with hard-to-compute kernel.

To run the, PyQuil is needed, plus a user code which can be gotten by siging up to use the Rigetti simulator online:
pip install pyquil
Also the standard packages, numpy, matplotlip, etc.
---------------------------------------------------------------------------------------------
MAXIMUM MEAN DISCREPANCY
----------------------------------------------------------------------------------------------
mmd.py contains the functions required to compute the cost function for the MMD, including its gradient when 
        applied to the quantum circuit. It computes the kernel, and the encoding function to encode the samples in 
         the quantum circuit for the hard-to-compute kernel. It also contains functions to compute all quantities exactly,
         and approximately.
        
      
classical_kernel.py computes Gaussian Kernel and outputs a versions as an numpy array, and a dictionary in the case of the 'Exact' 
                    version.

file_operations_in.py and file_operations_out.py have to be run to import the data, and kernel 
which can be precomputed for a given number of Epochs, and printed to a file. The data and kernel are import from the same file as dictionaries.

mmd_train_plot.py contains functions to print output data to a file, including the Born Machine parameters at each epoch, and also 
                  plots the MMD as a function of the number of Epochs.

sample_gen.py contains functions to produce a specified number of samples from the Born Machine, and the parameter shifted circuits   
              involved in computing the gradient.
              
kernel_circuit.py runs the  quantum circuit used for the hard-to-compute kernel, with two samples given as input. this is run as 
                  part of mmd.py when computing the gradient, for each of the samples required for the expectation value of the
                  MMD cost function gradient.

train_generation.py produces the training data, in a number of ways. It has functions to compute it exactly for the specified 
                    of qubits, and also a sampler to generate random samples according to the exact distribution. This is used 
                    when training the Born Machine approximately using samples.
          
param_init.py initialises the Born Machine circuit, with optional parameters to specify which circuit should be run, IQP, QAOA, or 
              IQPy. It also can run the parameter shifted versions of the circuits required to compute the MMD cost gradient.
              Finally, it initialises the Born Machine parameters at random, which ae to be trained.
              
run_and_compare.py calls mmd_train_plot a number of times to compare the output with various parameters which can be chosen.
---------------------------------------------------------------------------
KULLBACK LEIBLER DIVERGENCE
---------------------------------------------------------------------------
The initilisation codes for the Born Machine circuit are the same as with the MMD,
specifically:
param_init.py
train_generation.py

kl_div.py is the main file to compute the KL divergence, and the derivative, calling on
amplitude_computation_hadamard_test.py to run the hadamard test circuit, used to compute the amplitudes of the gradient of the KL cost function.

kl_run_and_compare.py runs and plots the KL divergence as a function of the Epoch number, subject
to a number of optional parameters. This can be done a specified number of times
