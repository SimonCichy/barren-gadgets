# TODO: make that into a class

# Necessary imports
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
# import datetime

from gadget_gradients_utils import hardware_efficient_ansatz, cat_state_preparation

# Automation from Hcomp to Hgad with decompose_hamiltonian(H[, hide_identity, …])

# Circuits preparing the state
def gadget_circuit(params, gate_sequence, computational_qubits, observable):       # also working for k'-local 
    assert(len(np.shape(params)) == 2)                  # check proper dimensions of params
    total_qubits = np.shape(params)[1]
    # for qubit in range(computational_qubits):
    # qml.PauliX(wires=[0])                       # cheating by preparing the |10...0> state
    if total_qubits > computational_qubits:
        cat_state_preparation(ancillary_qubits = range(computational_qubits, total_qubits, 1))
    hardware_efficient_ansatz(params, gate_sequence, rot_y=False)
    return qml.expval(observable)


# Cost functions
# TODO: rewrite to have only one cost function that receives an external observable -> observable to measure defined in the notebook
def computational_cost_function(params, gate_sequence, computational_qubits, device):
    gadget_qnode = qml.QNode(gadget_circuit, device)
    Hcomp = qml.PauliZ(0)
    for qubit in range(computational_qubits-1):
        Hcomp = Hcomp @ qml.PauliZ(qubit + 1)
    return gadget_qnode(params, gate_sequence, computational_qubits, Hcomp)


def gadget_cost_function(params, gate_sequence, computational_qubits, device, lambda_value):
    ancillary_terms = ancillary_cost_function(params, gate_sequence, computational_qubits, device)
    perturbation_terms = perturbation_cost_function(params, gate_sequence, computational_qubits, device)
    return ancillary_terms + lambda_value * perturbation_terms


def ancillary_cost_function(params, gate_sequence, computational_qubits, device):
    gadget_qnode = qml.QNode(gadget_circuit, device)
    assert(len(np.shape(params)) == 2)                  # check proper dimensions of params
    total_qubits = np.shape(params)[1]
    expectation_value = 0
    # creating the "unperturbed Hamiltonian"
    # acting on the ancillary qubits only
    for first_qubit in range(computational_qubits, total_qubits):
        for second_qubit in range(first_qubit+1, total_qubits):
            coeffs = [0.5, -0.5]
            obs = [qml.Identity(first_qubit) @ qml.Identity(second_qubit), qml.PauliZ(first_qubit) @ qml.PauliZ(second_qubit)]
            Hanc = qml.Hamiltonian(coeffs, obs)
            expectation_value += gadget_qnode(params, gate_sequence, computational_qubits, Hanc)
    return expectation_value


def perturbation_cost_function(params, gate_sequence, computational_qubits, device):
    gadget_qnode = qml.QNode(gadget_circuit, device)
    assert(len(np.shape(params)) == 2)                  # check proper dimensions of params
    expectation_value = 0
    # creating the perturbation part of the Hamiltonian
    # acting on both ancillary and target qubits with the same index
    for qubit in range(computational_qubits):           # /!\ only valid for 2-local
        V = qml.PauliZ(qubit) @ qml.PauliX(computational_qubits+qubit)
        expectation_value +=  gadget_qnode(params, gate_sequence, computational_qubits, V)
    return expectation_value


