import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np

from observables_holmes import ObservablesHolmes
from faehrmann_gadgets import NewPerturbativeGadgets

# For reproduceability and shareability
np.random.seed(42)

# Parameters of the tests
computational_qubits = 4
perturbation_factor = 1
locality = computational_qubits

# Used observables
oH = ObservablesHolmes(computational_qubits, 0, perturbation_factor)
Hcomp = oH.computational()
    

def test1():
    """Inspecting the terms of the gadgetized Hamiltonian"""
    gadgetizer = NewPerturbativeGadgets(perturbation_factor=perturbation_factor)
    print("Target computational Hamoltonian: ")
    print(Hcomp)
    print("Resulting gadgetized Hamiltonian: ")
    print(gadgetizer.gadgetize(Hcomp, target_locality=3))

def test2(): 
    """Checking the block diagonalization in terms of eigenstates of X"""
    Xoperator = qml.operation.Tensor(*[qml.PauliX(q) for q in range(locality)])
    # print(qml.matrix(Hcomp))
    # print(qml.matrix(Xoperator))


if __name__ == "__main__":
    test1()
    # test2()
