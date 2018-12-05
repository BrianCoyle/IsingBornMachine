from pyquil.quil import Program
from pyquil.paulis import *
import pyquil.paulis as pl
from pyquil.gates import *
from pyquil.api import get_qc
import pyquil as pyq
import numpy as np
import random as rand
from numpy import pi,log2


#Initialise Quantum State created after application of gate sequence
def TestFunc(device_name: str = "Aspen-1-2Q-B") -> None:
    print(pyq.list_quantum_computers())
    qc = get_qc(device_name)		
    qubes = qc.qubits()
    prog = Program()

    prog.inst(H(qubes[0]), CNOT(qubes[0], qubes[1]))

    meas_results_all_qubits = qc.run_and_measure(prog, 10)
    meas_results = np.vstack(meas_results_all_qubits[q] for q in sorted(qc.qubits())).T

    return prog, meas_results

prog, meas_results = TestFunc()
print(prog)
print(meas_results)