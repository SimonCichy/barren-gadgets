import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np

from hardware_efficient_ansatz import HardwareEfficientAnsatz
from observables_holmes import ObservablesHolmes
from gadget_cost import GadgetCost

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
Hloc = oH.local()
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


def test3():
    print("Testing the combination HardwareEfficientAnsatz + ExpvalCost" + 
          " on the computational and local Hamiltonians")
    print("Minimum eigenvalue of Hcomp:                       ", min(qml.eigvals(Hcomp)))
    print("Minimum eigenvalue of Hloc:                        ", min(qml.eigvals(Hloc)))
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits+ancillary_qubits)] for _ in range(num_layers)]
    hea = HardwareEfficientAnsatz(random_gate_sequence, initial_y_rot=False)
    cost_comp = qml.ExpvalCost(hea.ansatz, Hcomp, dev_gad)
    cost_loc = qml.ExpvalCost(hea.ansatz, Hloc, dev_gad)
    weights_init = np.zeros((num_layers, computational_qubits+ancillary_qubits), requires_grad=True)           # starting close to the ground state
    print("Expectation value of Hcomp with zero state:        ", cost_comp(weights_init))
    print("Expectation value of Hloc with zero state:         ", cost_loc(weights_init))
    weights = np.copy(weights_init)
    index = random_gate_sequence[0].index(qml.RX)
    print("Flipping qubit ", index)
    weights[0][index] = np.pi
    print("Expectation value of Hcomp with one flipped qubit: ", cost_comp(weights))
    print("Expectation value of Hloc with one flipped qubit:  ", cost_loc(weights))


def test4():
    print("Testing the combination HardwareEfficientAnsatz + ExpvalCost" + 
          " on the gadget Hamiltonian")
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits+ancillary_qubits)] for _ in range(num_layers)]
    hea = HardwareEfficientAnsatz(random_gate_sequence, initial_y_rot=False)
    cost_gad = qml.ExpvalCost(hea.ansatz, Hgad, dev_gad)
    cost_anc = qml.ExpvalCost(hea.ansatz, Hanc, dev_gad)
    cost_pert = qml.ExpvalCost(hea.ansatz, V, dev_gad)
    weights_init = np.zeros((num_layers, computational_qubits+ancillary_qubits), requires_grad=True)           # starting close to the ground state
    print("Expectation value of Hgad with zero state:         ", cost_gad(weights_init))
    print("Expectation value of Hanc with zero state:         ", cost_anc(weights_init))
    print("Expectation value of V with zero state:            ", cost_pert(weights_init))
    weights = weights_init
    index = random_gate_sequence[0].index(qml.RX)
    print("Flipping qubit ", index)
    weights[0][index] = np.pi
    print("Expectation value of Hgad with one flipped qubit:  ", cost_gad(weights))
    print("Expectation value of Hanc with one flipped qubit:  ", cost_anc(weights))
    print("Expectation value of V with one flipped qubit:     ", cost_pert(weights))


def test5():
    print("2-local decomposition of 4 qubits")
    print(ObservablesHolmes(4, 4, perturbation_factor).gadget())
    print("3-local decomposition of 4 qubits")
    print(ObservablesHolmes(4, 2, perturbation_factor).gadget())
    print("3-local decomposition of 6 qubits")
    print(ObservablesHolmes(6, 3, perturbation_factor).gadget())


if __name__ == "__main__":
    # test1()
    # test2()
    test3()
    test4()
    test5()
    