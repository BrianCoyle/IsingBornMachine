# IsingBornMachine
Implementation of Quantum Ising Born Machine using Rigetti Forest Platform, with two approaches: 
1. Computing the KL Divergence, 
2. MMD as a cost function with hard-to-compute kernel.

mmd.py contains the functions required to compute the cost function for the MMD, including its gradient when 
        applied to the quantum circuit. It computes the kernel, and the encoding function to encode the samples in 
         the quantum circuit for the hard-to-compute kernel. It also contains functions to compute all quantities exactly,
         and approximately.
        
      
classical_kernel.py computes Gaussian Kernel and outputs a versions as an numpy array, and a dictionary in the case of the 'Exact' 
                    version.
                    
mmd_plot.py contains functions to print output data to a file, including the Born Machine parameters at each epoch, and also plot
            the MMD as a function of the number of Epochs.

sample_gen.py contains functions to produce a specified number of samples from the Born Machine, and the parameter shifted circuits   
              involved in computing the gradient.
              
kernel_circuit.py runs the  quantum circuit used for the hard-to-compute kernel, with two samples given as input. 

train_generation.py produces the training data, in a number of ways. It has functions to compute it exactly for the specified 
                    of qubits, and also a sampler to generate random samples according to the exact distribution. This is used 
                    when training the Born Machine approximately using samples.
          
param_init.py initialises the Born Machine circuit, with optional parameters to specify which circuit should be run, IQP, QAOA, or 
              IQPy. It also can run the parameter shifted versions of the circuits required to compute the MMD cost gradient.
              Finally, it initialises the Born Machine parameters at random, which ae to be trained.
