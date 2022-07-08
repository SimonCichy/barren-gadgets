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
computational_qubits = 6
perturbation_factor = 1
locality = computational_qubits

# Used observables
oH = ObservablesHolmes(computational_qubits, 0, perturbation_factor)
Hcomp = oH.computational()
gadgetizer = NewPerturbativeGadgets(perturbation_factor=perturbation_factor)
    

def test1():
    """Inspecting the terms of the gadgetized Hamiltonian"""
    print("Target computational Hamoltonian: ")
    print(Hcomp)
    print("Resulting gadgetized 3-local Hamiltonian: ")
    print(gadgetizer.gadgetize(Hcomp, target_locality=3))
    print("Resulting gadgetized 4-local Hamiltonian: ")
    print(gadgetizer.gadgetize(Hcomp, target_locality=4))

def test2(): 
    """Inspecting the terms of the gadgetized Hamiltonian"""
    term1 = qml.operation.Tensor(*[qml.PauliZ(q) for q in range(computational_qubits)])
    term2 = qml.operation.Tensor(*[qml.PauliX(q) for q in range(computational_qubits)])
    Hcomp = qml.Hamiltonian([1,1], [term1, term2])
    print("Target computational Hamoltonian: ")
    print(Hcomp)
    print("Resulting gadgetized Hamiltonian: ")
    print(gadgetizer.gadgetize(Hcomp, target_locality=3))

def test3():
    """Exploring the norm of the gadget Hamiltonian"""
    Hgad = gadgetizer.gadgetize(Hcomp, target_locality=3)
    print(np.sum(np.abs(Hgad.coeffs)))


if __name__ == "__main__":
    test1()
    # test2()
    # test3()
