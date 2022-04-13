import pennylane as qml
from pennylane import numpy as np


class HardwareEfficientAnsatz:
    def __init__(self, gate_sequence=None, initial_y_rot=True, cat_range=None):
        self.gate_sequence = gate_sequence
        self.do_y = initial_y_rot
        self.cat_range = cat_range
    
    def generate_sequence(self, n_qubits, n_layers):
        gate_set = [qml.RX, qml.RY, qml.RZ]
        random_gate_sequence = [[np.random.choice(gate_set) for _ in range(n_qubits)] for _ in range(n_layers)]
        self.gate_sequence = random_gate_sequence
    
    def cat_state_preparation(self, wires):
        # leave the computational qubits in the |0> state
        # create a cat state |+> in the ancillary register
        # /!\ inefficient in depth
        qml.Hadamard(wires=wires[0])
        for qubit in wires[1:]:
            qml.CNOT(wires=[wires[0], qubit])
    
    def ansatz(self, params, wires):
        assert(len(np.shape(params)) == 2)      # check proper dimensions of params
        num_layers = np.shape(params)[0]        # np.shape(params) = (num_layers, num_qubits)
        num_qubits = np.shape(params)[1]
        assert(len(wires) == num_qubits)
        if self.gate_sequence == None:
            self.generate_sequence(num_qubits, num_layers)
        if self.do_y:
            for i in wires:
                qml.RY(np.pi/4, wires=i)
        if self.cat_range is not None:
            self.cat_state_preparation(self.cat_range)
        for l in range(num_layers):
            # Single random gate layer (single qubit rotations)
            for i in wires:
                self.gate_sequence[l][i](params[l][i], wires=i)
            # Nearest neighbour controlled phase gates
            if num_qubits > 1:                          # no entangling gates if using a single qubit
                qml.broadcast(qml.CZ, wires=wires, pattern="ring")

