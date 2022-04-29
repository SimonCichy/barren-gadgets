import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np

from observables_holmes import ObservablesHolmes
from jordan_gadgets import PerturbativeGadgets

# For reproduceability and shareability
np.random.seed(42)

# Parameters of the tests
computational_qubits = 4
ancillary_qubits = computational_qubits
total_qubits = computational_qubits + ancillary_qubits
perturbation_factor = 0.5
locality = computational_qubits
# lambda_max = (locality - 1) / (4 * locality)
# lambda_value = perturbation_factor * lambda_max

# Pennylane devices
dev_gad = qml.device("default.qubit", wires=range(computational_qubits+ancillary_qubits))

# Used observables
oH = ObservablesHolmes(computational_qubits, ancillary_qubits, perturbation_factor)
Hcomp = oH.computational()

    

def test1():
    """Inspecting the terms of the gadgetized Hamiltonian"""
    gadgetizer = PerturbativeGadgets(method='Jordan', perturbation_factor=perturbation_factor)
    print("Target computational Hamoltonian: ")
    print(Hcomp)
    print("Resulting gadgetized Hamiltonian: ")
    print(gadgetizer.gadgetize(Hcomp, target_locality=2))

def test2(): 
    """Checking the block diagonalization in terms of eigenstates of X"""
    Xoperator = qml.operation.Tensor(*[qml.PauliX(q) for q in range(locality)])
    # print(qml.matrix(Hcomp))
    # print(qml.matrix(Xoperator))


if __name__ == "__main__":
    test1()
    test2()
