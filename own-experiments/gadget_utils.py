import pennylane as qml
from pennylane import numpy as np

np.random.seed(42)


def hardware_efficient_ansatz(params, gate_sequence=None):
    """A random variational quantum circuit based on the hardware efficient ansatz. 
    There are no measurements and it is to be used within the global or local circuits

    Args:
        params (array[array[float]]): array of parameters of dimension (num_layers, num_qubits) containing the rotation angles

    Returns:
        not sure... see pennylane documentation
    """

    # Relevant parameters
    assert(len(np.shape(params)) == 2)      # check proper dimensions of params
    num_layers = np.shape(params)[0]        # np.shape(params) = (num_layers, num_qubits)
    num_qubits = np.shape(params)[1]
    
    if gate_sequence == None:
        # Generating the gate sequence from randomly applying RX, RY or RZ with the corresponding rotation angle
        gate_set = [qml.RX, qml.RY, qml.RZ]
        random_gate_sequence = [[np.random.choice(gate_set) for _ in range(num_qubits)] for _ in range(num_layers)]
        gate_sequence = random_gate_sequence

    # Initial rotations on all qubits
    for i in range(num_qubits):             # rotate all qubits
        qml.RY(np.pi / 4, wires=i)          # "to prevent X, Y , or Z from being an especially preferential 
                                            # direction with respect to gradients."

    # Repeating a layer structure
    for l in range(num_layers):
        # Single random gate layer (single qubit rotations)
        for i in range(num_qubits):
            gate_sequence[l][i](params[l][i], wires=i)
        # Nearest neighbour controlled phase gates
        if num_qubits > 1:                          # no entangling gates if using a single qubit
            qml.broadcast(qml.CZ, wires=range(num_qubits), pattern="ring")


def global_circuit(params):
    assert(len(np.shape(params)) == 2)      # check proper dimensions of params
    num_qubits = np.shape(params)[1]        # np.shape(params) = (num_layers, num_qubits)

    hardware_efficient_ansatz(params)
    # Objective operator H = Z_1 Z_2 ... Z_n
    H = qml.PauliZ(0)
    for qubit in range(num_qubits-1):
        H = H @ qml.PauliZ(qubit + 1)
    return qml.expval(H)


def local_circuit(params):
    hardware_efficient_ansatz(params)
    # Objective operator H = Z_1 Z_2
    return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))


def gadget2_circuit(params, term, target_qubits):
    assert(len(np.shape(params)) == 2)
    total_qubits = np.shape(params)[1]
    computational_qubits = int(total_qubits / 2)

    cat_state_preparation(ancillary_qubits=range(computational_qubits, total_qubits, 1))

    hardware_efficient_ansatz(params)

    # creating the "unperturbed Hamiltonian": Hanc
    if term == 'ancillary':
        # terms of the form I(aa)-Z(a)Z(a)
        term = qml.Identity(target_qubits[0]) @ qml.Identity(target_qubits[1]) - qml.PauliZ(target_qubits[0]) @ qml.PauliZ(target_qubits[1])
    elif term == 'coupling':
        # terms of the form Z(c)X(a)
        term = qml.PauliZ(target_qubits[0]) @ qml.PauliX(computational_qubits+target_qubits[0])
    return qml.expval(term)


def gadget3_circuit(params, term, target_qubits):
    assert len(np.shape(params)) == 2
    total_qubits = np.shape(params)[1]
    computational_qubits = int(total_qubits * 2 / 3)

    cat_state_preparation(ancillary_qubits=range(computational_qubits, total_qubits, 1))

    hardware_efficient_ansatz(params)

    # creating the "unperturbed Hamiltonian": Hanc
    if term == 'ancillary':
        # terms of the form I(aa)-Z(a)Z(a)
        term = qml.Identity(target_qubits[0]) @ qml.Identity(target_qubits[1]) - qml.PauliZ(target_qubits[0]) @ qml.PauliZ(target_qubits[1])
    elif term == 'coupling':
        # terms of the form Z(c)Z(c)X(a)
        term = qml.PauliZ(target_qubits[0]) @ qml.PauliZ(target_qubits[1]) @ qml.PauliX(computational_qubits+target_qubits[2])
    return qml.expval(term)


def cost_function(qnode, params, circuit_type, num_qubits=0, lam=1):
    if circuit_type == gadget2_circuit:
        assert(num_qubits!=0)
        # Objective operator H = Hanc + V
        computational_qubits = num_qubits
        ancillary_qubits = computational_qubits
        total_qubits = ancillary_qubits + computational_qubits
        expval_terms = []
        # creating the "unperturbed Hamiltonian"
        # acting on the ancillary qubits only
        for first_qubit in range(computational_qubits, total_qubits):
            for second_qubit in range(first_qubit+1, total_qubits):
                expval_terms.append(qnode(params, term='ancillary', target_qubits=[first_qubit, second_qubit]))
        # creating the perturbation part of the Hamiltonian
        # acting on both ancillary and target qubits with the same index
        for qubit in range(computational_qubits):
            expval_terms.append(qnode(params, term='coupling', target_qubits=[qubit]))
        
        return 0.5 * np.sum(expval_terms[:-num_qubits]) + lam * np.sum(expval_terms[-num_qubits:])
    elif circuit_type == gadget3_circuit:
        assert num_qubits != 0
        assert num_qubits % 2 == 0, "3-local gadget decomposition only implemented for even qubit numbers"
        # Objective operator H = Hanc + V
        computational_qubits = num_qubits
        ancillary_qubits = int(computational_qubits  / 2)
        total_qubits = ancillary_qubits + computational_qubits
        expval_terms = []
        # creating the "unperturbed Hamiltonian"
        # acting on the ancillary qubits only
        for first_qubit in range(computational_qubits, total_qubits):
            for second_qubit in range(first_qubit+1, total_qubits):
                expval_terms.append(qnode(params, term='ancillary', target_qubits=[first_qubit, second_qubit]))
        # creating the perturbation part of the Hamiltonian
        # acting on both ancillary and target qubits
        for qubit in range(ancillary_qubits):
            target_qubits = [2*qubit, 2*qubit+1, qubit]
            expval_terms.append(qnode(params, term='coupling', target_qubits=target_qubits))
        
        return 0.5 * np.sum(expval_terms[:-num_qubits]) + lam * np.sum(expval_terms[-num_qubits:])
    else:
        return qnode(params)                # the cost function is the expectation value


def cat_state_preparation(ancillary_qubits):
    # leave the computational qubits in the |0> state
    # create a cat state |+> in the ancillary register
    # /!\ inefficient in depth
    qml.Hadamard(wires=ancillary_qubits[0])
    for qubit in ancillary_qubits[1:]:
        qml.CNOT(wires=[ancillary_qubits[0], qubit])





