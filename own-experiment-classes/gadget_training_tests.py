import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np
# from gadget_training_utils import *
from gadget_cost import GadgetCost
from observables_holmes import ObservablesHolmes

# For reproduceability and shareability
np.random.seed(42)

# Parameters of the simulation
computational_qubits = 4
ancillary_qubits = computational_qubits
total_qubits = computational_qubits + ancillary_qubits
num_layers = 2
max_iter = 50
# num_samples = 200
perturbation_factor = 0.5
locality = computational_qubits
lambda_max = (locality - 1) / (4 * locality)
lambda_value = perturbation_factor * lambda_max
gate_set = [qml.RX, qml.RY, qml.RZ]

# Pennylane devices
dev_comp = qml.device("default.qubit", wires=range(computational_qubits))
dev_gad = qml.device("default.qubit", wires=range(computational_qubits+ancillary_qubits))

# Used observables
oH = ObservablesHolmes(computational_qubits, ancillary_qubits, perturbation_factor)
Hcomp = oH.computational()
Hanc = oH.ancillary()
V = oH.perturbation()
Hgad = oH.gadget()

# creating a cost function object
cf = GadgetCost(computational_qubits, total_qubits, dev_gad)

def test1():
    """ Test 1:
    # All weights to 0 -> (not anymore) probed state |10...0>|+>
    # Expected results: 
    # - ancillary cost = 0
    # - perturbation cost = lambda
    # - (not anymore) computational cost = -1 """
    weights_init = np.zeros((num_layers, computational_qubits+ancillary_qubits), requires_grad=True)           # starting close to the ground state
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits+ancillary_qubits)] for _ in range(num_layers)]
    print(cf.cost_function(weights_init, random_gate_sequence, Hanc) == 0)
    print(cf.cost_function(weights_init, random_gate_sequence, V) == 0)
    print(cf.cost_function(weights_init, random_gate_sequence, Hgad) == 0)
    print(cf.cost_function(weights_init, random_gate_sequence, Hcomp))
    


def test2():
    """Test 2: Same eigenvectors"""
    print(Hcomp)
    print(min(qml.eigvals(Hcomp)))
    print(Hgad)
    # print(min(np.linalg.eig(qml.matrix(Hgad))[0]))
    # print(np.linalg.eig(qml.matrix(Hgad)))
    print(min(qml.eigvals(Hgad)))


if __name__ == "__main__":
    test1()
    test2()
    