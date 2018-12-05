# IsingBornMachine
Implementation of Quantum Ising Born Machine using Rigetti Forest Platform, with two approaches: 
1. Training using MMD as a Cost function.
2. Training using Stein Discrepancy as a Cost function

Both of the above have the option to run using a 'Quantum-Hard' Kernel function

To run the, PyQuil is needed, plus a user code which can be gotten by signing up to use the Rigetti simulator online:

Follow instructions online:
http://docs.rigetti.com/en/stable/start.html
This will allow you to download the Rigetti SDK which includes a compiler and a Quantum Virtual Machine

Also the standard packages, numpy, matplotlib, etc.

In the current form of the above codes, it is necessary to run some functions in
file_operations_out.py to generate the data and pre-compute the quantum kernel etc.

---------------------------------------------------------------------------------------------
MAXIMUM MEAN DISCREPANCY
----------------------------------------------------------------------------------------------

---------------------------------------------------------------------------------------------
INSTRUCTIONS FOR USE
---------------------------------------------------------------------------------------------

run using:

```python
python run_and_compare.py input.txt
```

Where input.txt should look like:

N_epochs

learning_rate_one

learning_rate_two

N_data_samples

N_kernel_samples

batch_size

kernel_type

approx

cost_func

stein_approx

weight_sign
