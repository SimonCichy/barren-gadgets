# Necessary imports
import pennylane as qml
from pennylane import numpy as np


def hardware_efficient_ansatz(params, wires, gate_sequence=None, rot_y=True):
    # Relevant parameters
    assert(len(np.shape(params)) == 2)      # check proper dimensions of params
    num_layers = np.shape(params)[0]        # np.shape(params) = (num_layers, num_qubits)
    num_qubits = np.shape(params)[1]
    assert(len(wires) == num_qubits)
    
    if gate_sequence == None:
        # Generating the gate sequence from randomly applying RX, RY or RZ with the corresponding rotation angle
        print('genenrating sequence')
        gate_set = [qml.RX, qml.RY, qml.RZ]
        random_gate_sequence = [[np.random.choice(gate_set) for _ in range(num_qubits)] for _ in range(num_layers)]
        gate_sequence = random_gate_sequence

    # Initial rotations on all qubits
    if rot_y:
        print('rotating')
        for i in wires:                         # rotate all qubits
            qml.RY(np.pi / 4, wires=i)          # "to prevent X, Y , or Z from being an especially preferential 
                                                # direction with respect to gradients."

    # Repeating a layer structure
    for l in range(num_layers):
        # Single random gate layer (single qubit rotations)
        for i in wires:
            gate_sequence[l][i](params[l][i], wires=i)
        # Nearest neighbour controlled phase gates
        if num_qubits > 1:                          # no entangling gates if using a single qubit
            qml.broadcast(qml.CZ, wires=wires, pattern="ring")


def cat_state_preparation(wires):
    # leave the computational qubits in the |0> state
    # create a cat state |+> in the ancillary register
    # /!\ inefficient in depth
    qml.Hadamard(wires=wires[0])
    for qubit in wires[1:]:
        qml.CNOT(wires=[wires[0], qubit])
