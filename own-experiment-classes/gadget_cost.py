# Necessary imports
import pennylane as qml
from pennylane import numpy as np


def hardware_efficient_ansatz(params, gate_sequence=None, rot_y = True):
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
    if rot_y:
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


def cat_state_preparation(ancillary_qubits):
    # leave the computational qubits in the |0> state
    # create a cat state |+> in the ancillary register
    # /!\ inefficient in depth
    qml.Hadamard(wires=ancillary_qubits[0])
    for qubit in ancillary_qubits[1:]:
        qml.CNOT(wires=[ancillary_qubits[0], qubit])


class GadgetCost:
    def __init__(self, computational_qubits, total_qubits, device):
        self.n_comp = computational_qubits
        self.n_tot = total_qubits
        self.dev = device

    # Circuits preparing the state
    def gadget_circuit(self, params, gate_sequence, observable):       # also working for k'-local 
        assert(len(np.shape(params)) == 2)                  # check proper dimensions of params
        assert (self.n_tot == np.shape(params)[1])
        # for qubit in range(computational_qubits):
        # qml.PauliX(wires=[0])                       # cheating by preparing the |10...0> state
        # if self.n_tot > self.n_comp:
        #     cat_state_preparation(ancillary_qubits = range(self.n_comp, self.n_tot, 1))
        hardware_efficient_ansatz(params, gate_sequence, rot_y=False)
        return qml.expval(observable)
    
    def cost_function(self, params, gate_sequence, observable):
        gadget_qnode = qml.QNode(self.gadget_circuit, self.dev)
        return gadget_qnode(params, gate_sequence, observable)



