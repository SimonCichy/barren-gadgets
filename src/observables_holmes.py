import pennylane as qml
from pennylane import numpy as np


class ObservablesHolmes:
    def __init__(self, computational_qubits, ancillary_qubits, perturbation_factor):
        self.n_comp = int(computational_qubits)
        self.n_anc = int(ancillary_qubits)
        self.n_tot = int(self.n_comp + self.n_anc)
        self.factor = perturbation_factor
        self.l = perturbation_factor * (self.n_comp - 1) / (4 * self.n_comp)
        if self.n_anc != 0:
            assert self.n_comp % self.n_anc == 0, "computational qubits not divisible by ancillary qubits. Non-divisible decomposition not implemented yet"
            self.loc = int(self.n_comp / self.n_anc)
    
    def computational(self):
        Hcomp = qml.PauliZ(0)
        for qubit in range(self.n_comp-1):
            Hcomp = Hcomp @ qml.PauliZ(qubit + 1)
        return qml.Hamiltonian([1], [Hcomp])
    
    def local(self):
        Hloc = qml.PauliZ(0) @ qml.PauliZ(1)
        return qml.Hamiltonian([1], [Hloc])

    def ancillary(self):
        coeffs = []
        obs = []
        for first_qubit in range(self.n_comp, self.n_tot):
            for second_qubit in range(first_qubit+1, self.n_tot):
                coeffs += [0.5, -0.5]
                obs += [qml.Identity(first_qubit) @ qml.Identity(second_qubit), qml.PauliZ(first_qubit) @ qml.PauliZ(second_qubit)]
        Hanc = qml.Hamiltonian(coeffs, obs)
        return Hanc
    
    # def perturbation(self):
    #     coeffs = [self.l] * self.n_comp
    #     obs = []
    #     for qubit in range(self.n_comp):           # /!\ only valid for 2-local
    #         obs += [qml.PauliZ(qubit) @ qml.PauliX(self.n_comp+qubit)]
    #     V = qml.Hamiltonian(coeffs, obs)
    #     return V
    
    def perturbation(self):
        coeffs = [self.l] * self.n_anc
        obs = []
        for anc_qubit in range(self.n_anc):           # /!\ only valid for 2-local
            term = qml.PauliX(self.n_comp+anc_qubit)
            for q in range(self.loc):
                term = term @ qml.PauliZ(self.loc*anc_qubit+q)
            # obs += [qml.PauliZ(anc_qubit) @ qml.PauliX(self.n_comp+anc_qubit)]
            obs += [term]
        V = qml.Hamiltonian(coeffs, obs)
        return V
    
    def gadget(self):
        Hgad = self.ancillary() + self.perturbation()
        return Hgad

    def cat_projector(self):
        cat_state = np.zeros(2**self.n_anc)
        cat_state[[0, -1]] = 1/np.sqrt(2) 
        cat_projector = qml.Hermitian(np.outer(cat_state, cat_state), range(self.n_comp, self.n_tot, 1))
        return cat_projector

    def ancillary_ground_projector(self):
        projector = np.zeros((2**self.n_anc, 2**self.n_anc))
        projector[[0, -1],[0, -1]] = 1
        projector = qml.Hermitian(projector, range(self.n_comp, self.n_tot, 1))
        return projector
