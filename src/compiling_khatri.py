from threading import local
from sympy.utilities.iterables import multiset_permutations
import pennylane as qml
from pennylane import numpy as np


class CompilingKhatri:
    """ Class to generate the trainable circuits used in the quantum compiling
    paper khatri2019 when looking at larger scale problems (Section ??)
     
    
    Args: 
        computational_qubits (int)  : number of qubits acted upon by the 
                                      target unitary to be compiled
    """
    def __init__(self, computational_qubits, example):
        self.n_comp = int(computational_qubits)
        self.n_tot = int(2 * self.n_comp)
        self.example = example
        assert example in [1, 2]
        self.params_ex = [None, None, None]
        self.params_ex[1] = np.random.uniform(0, np.pi, 
                          size=self.n_comp)
        self.params_ex[2] = np.random.uniform(0, np.pi, 
                          size=(2, self.n_comp))

    def cost_global(self, params):
        return 1 - self.circuit(params, range(self.n_comp))[0]
    
    def cost_local(self, params):
        fidelities = [self.circuit(params, target)[0] for target in range(self.n_comp)]
        return 1 - np.sum(fidelities)/self.n_comp

    def circuit(self, params, target_qubits):
        """Method to generate the circuit of the examples"""
        self.entangled_state_prep()
        if self.example == 1:
            self.example1(params)
        elif self.example == 2:
            self.example2(params)
        self.HST(target_qubits)
    
    def entangled_state_prep(self):
        """ Generate the maximally entangled state over all qubits"""
        # Hadamards on all computational qubits
        for qubit in range(self.n_comp):
            qml.Hadamard(wires=qubit)
        # CNOTs
        for qubit in range(self.n_comp):
            qml.CNOT(wires=[qubit, qubit+self.n_comp])

    def HST(self, target_qubits):
        """Hilbert Schmidt Test on all qubits"""
        for qubit in target_qubits:
            qml.CNOT(wires=[qubit, qubit+self.n_comp])
        for qubit in target_qubits:
            qml.Hadamard(wires=qubit)
        measured_qubits = target_qubits + \
                          [self.n_comp + q for q in target_qubits]
        return qml.probs(wires=measured_qubits)
    
    def example1(self, params):
        # target unitary U to be learned 
        for qubit in range(self.n_comp):
            qml.RZ(self.params_ex[1][qubit])
        # learned unitary V 
        for qubit in range(self.n_comp, self.n_tot, 1):
            qml.RZ(params[qubit])

    def example2(self, params):
        # target unitary U to be learned 
        for qubit in range(self.n_comp):
            qml.RZ(self.params_ex[2][0][qubit])
        #TODO check the shape of the chain broadcast
        qml.broadcast(qml.CZ, wires=range(self.n_comp), pattern="chain")
        for qubit in range(self.n_comp):
            qml.RZ(self.params_ex[2][1][qubit])
        # learned unitary V 
        for qubit in range(self.n_comp, self.n_tot, 1):
            qml.RZ(params[0][qubit])
        qml.broadcast(qml.CZ, wires=range(self.n_comp), pattern="chain")
        for qubit in range(self.n_comp):
            qml.RZ(params[1][qubit])



