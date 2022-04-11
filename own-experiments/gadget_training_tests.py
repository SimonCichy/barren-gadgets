from gadget_training_utils import *

# For reproduceability and shareability
np.random.seed(42)
# data_folder = '../../results/data/'

# Parameters of the simulation
computational_qubits = 2
ancillary_qubits = computational_qubits
num_layers = 2
max_iter = 50
# num_samples = 200
perturbation_factor = 0.5
locality = computational_qubits
lambda_max = (locality - 1) / (4 * locality)
gate_set = [qml.RX, qml.RY, qml.RZ]

# Pennylane devices
dev_comp = qml.device("default.qubit", wires=range(computational_qubits))
dev_gad = qml.device("default.qubit", wires=range(computational_qubits+ancillary_qubits))


if __name__ == "__main__":
    # Test 1:
    # All weights to 0 -> probed state |11...1>|+>
    # Expected results: 
    # - ancillary cost = 0
    # - perturbation cost = lambda
    weights_init = np.zeros((num_layers, computational_qubits+ancillary_qubits), requires_grad=True)           # starting close to the ground state
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits+ancillary_qubits)] for _ in range(num_layers)]
    # print(unperturbed_qnode(weights_init, random_gate_sequence, computational_qubits, [6, 7]))
    print(ancillary_cost_function(weights_init, random_gate_sequence, computational_qubits, dev_gad)==0)
    print(perturbation_cost_function(weights_init, random_gate_sequence, computational_qubits, dev_gad)==0)
    print(gadget_cost_function(weights_init, random_gate_sequence, computational_qubits, dev_gad, perturbation_factor*lambda_max)==0)
    print(computational_cost_function(weights_init, random_gate_sequence, computational_qubits, dev_gad))

    # Test 2: Same eigenvectors
    n = 2
    Hcomp = qml.PauliZ(0)
    for q in range(1, n, 1):
        Hcomp = Hcomp @ qml.PauliZ(q)
    print(Hcomp)
    # print(np.linalg.eig(qml.matrix(Hcomp)))
    for first_qubit in range(n, 2*n):
        for second_qubit in range(first_qubit+1, 2*n):
            coeffs = [0.5, -0.5]
            obs = [qml.Identity(first_qubit) @ qml.Identity(second_qubit), qml.PauliZ(first_qubit) @ qml.PauliZ(second_qubit)]
            if second_qubit == n+1:
                Hanc = qml.Hamiltonian(coeffs, obs)
            else: 
                Hanc += qml.Hamiltonian(coeffs, obs)
    V = qml.PauliZ(0) @ qml.PauliX(n)
    for qubit in range(1, n, 1):           # /!\ only valid for 2-local
        V += qml.PauliZ(qubit) @ qml.PauliX(n+qubit)
    Hgad = Hanc + V
    print(Hgad)
    print(np.linalg.eig(qml.matrix(Hgad)))