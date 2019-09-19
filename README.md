# IsingBornMachine
Implementation of Quantum Ising Born Machine using Rigetti Forest Platform, with two approaches: 
1. Training using MMD as a Cost function.
2. Training using the Stein Discrepancy as a Cost Function
3. Traing using the Sinkhorn Divergence as a Cost Function.
Both of the above have the option to run using a 'Quantum-Hard' Kernel function

To run the, PyQuil is needed, plus a user code which can be gotten by signing up to use the Rigetti simulator online:

Follow instructions online:
http://docs.rigetti.com/en/stable/start.html
This will allow you to download the Rigetti SDK which includes a compiler and a Quantum Virtual Machine

Also numpy, matplotlib are required etc.

Finally, pytorch will need to be installed to run the tensor operations in feydy_sink.py

---------------------------------------------------------------------------------------------
Maximum Mean Discrepancy (MMD)
----------------------------------------------------------------------------------------------
The Maximum Mean Discrepancy is a cost function used for two-sample testing. It compares a measure
of 'Discrepancy' between two distributions P and Q. If and only if the distributions are idential, 
the MMD = 0.

Here we wish to compare between the empirical distribution outputted by the Born Machine, and 
some data distribution, both of which are given as binary samples.

---------------------------------------------------------------------------------------------
Stein Discrepancy
----------------------------------------------------------------------------------------------
The Stein Discrepancy is an alternative cost function which may be used to compare distributions.
Typically, it is used as a 'goodness-of-fit' test, to determine whether samples come from a 
particular distribution or not. This is due to its asymmetry, whereas the MMD is symmetric in 
the distributions.

---------------------------------------------------------------------------------------------
Sinkhorn Divergence
----------------------------------------------------------------------------------------------

Finally, the Sinkhorn Divergence is a third cost function which leverages both the favourable 
qualities of the MMD, with the so-called Wasserstein Distance. The Wasserstein Distance is 
strongly related to the notion of 'Optimal Transport', or a means of moving between two 
distributions by minimising some 'cost'. The Wasserstein has the capability of taking into 
account the different between points in the distributions due to the use of this cost, which 
is taken to be a metric on the sample space. However, it is notoriously hard to estimate from 
samples, i.e. it has a sample complexity which scales exponentially with the size of the space.

The Sinkhorn Divergence is an attempt to call on the ease of computability of the MMD, but the 
favourable properties of the Wasserstein Distance. In fact, it is a version of Wasserstein which
is regularized by an entropy term, which makes the problem strongly convex. 

---------------------------------------------------------------------------------------------
INSTRUCTIONS FOR USE
---------------------------------------------------------------------------------------------

run using:

```shell
python3 run_and_compare.py inputs.txt
```

Where input.txt should look like:

N_epochs   

learning_rate

data_type

N_data_samples

N_born_samples

N_kernel_samples

batch_size

kernel_type

cost_func

device_name

as_qvm_value (1) = True, (0) = False

stein_score

stein_eigvecs

stein_eta

sinkhorn_eps

Where:

1. N_epochs is the number of epochs the training will run for.
2. data_type will define the type of data which is to be learned either:
  'Classical_Data', which is generated by a simple function or 
  'Quantum_Data', which is outputted by a quantum circuit
3. N_data_samples defines the number of data samples to be used from the data distribution 
                to learn from
4. N_born_samples defines the number of born samples to be used from the Ising Born Machine
              to determine the instantaneous distribution and to assess learning progress
5. N_kernel_samples defines the number of samples to be used when computing a single kernel element
                 If a Quantum kernel is chosen, the kernel will be the overlap between two states
                 which is determined as a result of measurements
6. batch_size is the number of samples to be used in each mini-batch during training
7. kernel_type is either 'Quantum' or 'Gaussian', depending on whether a mixture of Gaussians kernel or a 
               quantum kernel is used in the cost_function         
8. cost_func is the choice of cost function used for training, either 'MMD' or 'Stein'
9. device_name is the Rigetti chip to be used, it also determines the number of qubits to be used:
            e.g. Aspen-1-2Q-B uses a particular two qubits from the Aspen chip
10. as_qvm_value determines whether to run on the Rigetti simulator, or the actual Quantum chip
11. stein_score is the choice of method to compute the Stein Score function, either 'Exact_Score', 'Identity_Score' or           'Spectral_Score', to compute using exact probabilities, inverting Stein's identity, or the spectral method respectively
12. stein_eigvecs is the number of Nystrom eigenvectors required to compute the Stein Score using the Spectral Method, an      integer.
13. stein_eta is the regularisation parameter required in the Identity Score method, a small number, typically 0.01
14. sinkhorn_eps is the regularisation parameter used to compute the Sinkhorn Divergence, between (0, infinity)
--------------------------------------------------------------------------------------------
Generate Data & Kernels
--------------------------------------------------------------------------------------------
To Generate Classical (Mixture of Gaussians) kernels for all qubits up to 8:

```shell
python3 file_operations_out.py None Gaussian 8
```
To Generate Quantum kernels for all qubits up to 8:

```shell
python3 file_operations_out.py None Quantum 8
```
To Generate Classical Data (Mixture of Bernoulli Modes) for all qubits up to 8:

```shell
python3 file_operations_out.py Bernoulli None 8
```

To Generate Quantum Data (from a fully connected IQP circuit particularly) for all qubits up to 8:

```shell
python3 file_operations_out.py Quantum None 8
```
