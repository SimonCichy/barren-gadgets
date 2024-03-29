import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np

from hardware_efficient_ansatz import AlternatingLayeredAnsatz
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
Hloc = oH.local()
Hanc = oH.ancillary()
V = oH.perturbation()
Hgad = oH.gadget()
Pcat = oH.cat_projector()
Pground_comp = oH.computational_ground_projector()
Pground_anc = oH.ancillary_ground_projector()
    

def test2():
    evals = qml.eigvals(ObservablesHolmes(4, 4, perturbation_factor).gadget())
    print("Eigenvalues of the 2-local decomposition of 4 qubits: ")
    print(evals)
    print("Minimum eigenvalue:  ", min(evals))


def test3():
    print("Testing the combination AlternatingLayeredAnsatz + ExpvalCost" + 
          " on the computational and local Hamiltonians")
    print("Minimum eigenvalue of Hcomp:                       ", min(qml.eigvals(Hcomp)))
    print("Minimum eigenvalue of Hloc:                        ", min(qml.eigvals(Hloc)))
    random_gate_sequence = [[np.random.choice(gate_set) 
                             for _ in range(computational_qubits+ancillary_qubits)] 
                             for _ in range(num_layers)]
    ala = AlternatingLayeredAnsatz(random_gate_sequence, initial_y_rot=False)
    cost_comp = qml.ExpvalCost(ala.ansatz, Hcomp, dev_gad)
    cost_loc = qml.ExpvalCost(ala.ansatz, Hloc, dev_gad)
    cost_cat = qml.ExpvalCost(ala.ansatz, Pcat, dev_gad)
    cost_gs_comp = qml.ExpvalCost(ala.ansatz, Pground_comp, dev_gad)
    cost_gs_anc = qml.ExpvalCost(ala.ansatz, Pground_anc, dev_gad)
    weights_init = np.zeros((num_layers, computational_qubits+ancillary_qubits), requires_grad=True)           # starting close to the ground state
    print("Expectation value of Hcomp with zero state:        ", cost_comp(weights_init))
    print("Expectation value of Hloc with zero state:         ", cost_loc(weights_init))
    print("Projection on comp gs with zero state:             ", cost_gs_comp(weights_init))
    print("Projection on ancilla gs with zero state:          ", cost_gs_anc(weights_init))
    print("Projection on cat state with zero state:           ", cost_cat(weights_init))
    weights = np.copy(weights_init)
    index = random_gate_sequence[0].index(qml.RX, 0, computational_qubits)
    print("Flipping (computational) qubit ", index)
    weights[0][index] = np.pi
    print("Expectation value of Hcomp with one flipped qubit: ", cost_comp(weights))
    print("Expectation value of Hloc with one flipped qubit:  ", cost_loc(weights))
    print("Projection on comp gs with one flipped qubit:      ", cost_gs_comp(weights))
    print("Projection on ancilla gs with one flipped qubit:   ", cost_gs_anc(weights))
    print("Projection on cat state with one flipped qubit:    ", cost_cat(weights))
    weights = np.copy(weights_init)
    index = random_gate_sequence[-1].index(qml.RX, computational_qubits, total_qubits)
    print("Flipping (ancillary) qubit ", index)
    weights[-1][index] = np.pi
    print("Expectation value of Hcomp with one flipped qubit: ", cost_comp(weights))
    print("Expectation value of Hloc with one flipped qubit:  ", cost_loc(weights))
    print("Projection on comp gs with one flipped qubit:      ", cost_gs_comp(weights))
    print("Projection on ancilla gs with one flipped qubit:   ", cost_gs_anc(weights))
    print("Projection on cat state with one flipped qubit:    ", cost_cat(weights))


def test4():
    print("Testing the combination AlternatingLayeredAnsatz + ExpvalCost" + 
          " on the gadget Hamiltonian")
    random_gate_sequence = [[np.random.choice(gate_set) 
                            for _ in range(computational_qubits+ancillary_qubits)] 
                            for _ in range(num_layers)]
    ala = AlternatingLayeredAnsatz(random_gate_sequence, initial_y_rot=False)
    cost_gad = qml.ExpvalCost(ala.ansatz, Hgad, dev_gad)
    cost_anc = qml.ExpvalCost(ala.ansatz, Hanc, dev_gad)
    cost_pert = qml.ExpvalCost(ala.ansatz, V, dev_gad)
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

def test7():
    random_gate_sequence = [[np.random.choice(gate_set) 
                             for _ in range(computational_qubits+ancillary_qubits)] 
                             for _ in range(num_layers)]
    ala = AlternatingLayeredAnsatz(random_gate_sequence, initial_y_rot=False)
    @qml.qnode(dev_gad)
    def circuit(params):
        ala.ansatz(params, wires=range(total_qubits))
        return qml.state()
    weights_init = np.zeros((num_layers, computational_qubits+ancillary_qubits), 
                            requires_grad=True)
    # print(circuit(weights_init))
    weights = weights_init
    index = random_gate_sequence[-1].index(qml.RX)
    print("Flipping qubit ", index)
    weights[-1][index] = np.pi
    print(circuit(weights))
    print(random_gate_sequence[0][index])


if __name__ == "__main__":
    # test2()
    test3()
    # test4()
    # test5()
    # test7()

