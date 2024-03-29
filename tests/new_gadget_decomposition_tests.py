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

def test4():
    """Testing the reordering"""
    term1 = qml.operation.Tensor(*[qml.PauliZ(q) for q in range(computational_qubits)])
    term2 = qml.operation.Tensor(*[qml.PauliX(q) for q in range(computational_qubits)])
    Hcomp = qml.Hamiltonian([1,1], [term1, term2])
    Hgad = gadgetizer.gadgetize(Hcomp, target_locality=3)
    Hs = [
        qml.PauliX(4),
        qml.operation.Tensor(qml.PauliX(0), qml.Hadamard(2), qml.PauliZ(4)),
        qml.Hamiltonian(
            [1,1],
            [
                qml.operation.Tensor(qml.PauliZ(0), qml.PauliX(1), qml.PauliZ(2)),
                qml.operation.Tensor(qml.PauliX(0), qml.PauliZ(1), qml.PauliX(2)),
            ]
        ),
        Hgad
    ]
    wires_map = {i: i+10 for i in range(18)}
    for H in Hs:
        print(H)
        new_H = gadgetizer.map_wires(H, wires_map)
        print(new_H)

def test5():
    """Testing the reordering"""
    term1 = qml.operation.Tensor(*[qml.PauliZ(q) for q in range(computational_qubits)])
    term2 = qml.operation.Tensor(*[qml.PauliX(q) for q in range(computational_qubits)])
    Hcomp = qml.Hamiltonian([1,1], [term1, term2])
    Hgad = gadgetizer.gadgetize(Hcomp, target_locality=3)
    print(Hgad)
    wires_map = gadgetizer.get_qubit_mapping(Hcomp, Hgad)
    ordered_Hgad = gadgetizer.map_wires(Hgad, wires_map)
    print(ordered_Hgad)


if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    test5()
