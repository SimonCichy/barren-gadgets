# Necessary imports
import pennylane as qml
from pennylane import numpy as np

from gadget_gradients_utils import hardware_efficient_ansatz, cat_state_preparation


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



