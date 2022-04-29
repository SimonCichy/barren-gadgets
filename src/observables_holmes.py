from sympy.utilities.iterables import multiset_permutations
import pennylane as qml
from pennylane import numpy as np


class ObservablesHolmes:
    """ Class to generate the relevant observables (computational, local, gadget 
    decomposition and some projectors) for the implementation of perturbative 
    gadgets (Jordan2012) on one example for Barren plateaus (Holmes2021) 
    
    Args: 
        computational_qubits (int)  : number of qubits in the target 
                                      computational Hamiltonian
        ancillary_qubits (int)      : number of additional ancillary qubits to 
                                      use in the gadget decomposition (should 
                                      divide the number of computational qubits
                                      in the current implementation)
        perturbation_factor (float) : parameter controlling the magnitude of the
                                      perturbation (should be in [0, 1] and is a
                                      pre-factor to \lambda_max as in 
                                      Jordan2012"""
    def __init__(self, computational_qubits, ancillary_qubits, perturbation_factor):
        self.n_comp = int(computational_qubits)
        self.n_anc = int(ancillary_qubits)
        self.n_tot = int(self.n_comp + self.n_anc)
        self.factor = perturbation_factor
        self.l = perturbation_factor * (self.n_comp - 1) / (4 * self.n_comp)
        if self.n_anc != 0:
            assert self.n_comp % self.n_anc == 0, "computational qubits not divisible by ancillary qubits. Non-divisible decomposition not implemented yet"
            self.loc = int(self.n_comp / self.n_anc) + 1
    
    def computational(self):
        """Generation of the computational Hamiltonian of the Holmes2021 paper to be used as a cost function with qml.ExpvalCost
        Args: None
        Returns:
            observable (qml.Hamiltonian)    : computational Hamiltonian to be used as observable
        """
        # TODO: substitute with and check
        # Hcomp = qml.operation.Tensor(*[qml.PauliZ(q) for q in range(self.n_comp)])
        Hcomp = qml.PauliZ(0)
        for qubit in range(self.n_comp-1):
            Hcomp = Hcomp @ qml.PauliZ(qubit + 1)
        return qml.Hamiltonian([1], [Hcomp])
    
    def local(self):
        """Generation of the local Hamiltonian of the Holmes2021 paper to be used as a cost function with qml.ExpvalCost
        Args: None
        Returns:
            observable (qml.Hamiltonian)    : local Hamiltonian to be used as observable
        """
        Hloc = qml.PauliZ(0) @ qml.PauliZ(1)
        return qml.Hamiltonian([1], [Hloc])

    def ancillary(self):
        """Generation of the ancillary (unperturbed) part of the gadget 
        Hamiltonian corresponding to the computational Hamiltonian of the 
        Holmes2021 paper
        Args: None
        Returns:
            observable (qml.Hamiltonian)    : ancillary Hamiltonian
        """
        coeffs = []
        obs = []
        for first_qubit in range(self.n_comp, self.n_tot):
            for second_qubit in range(first_qubit+1, self.n_tot):
                coeffs += [0.5, -0.5]
                obs += [qml.Identity(first_qubit) @ qml.Identity(second_qubit), 
                        qml.PauliZ(first_qubit) @ qml.PauliZ(second_qubit)]
        Hanc = qml.Hamiltonian(coeffs, obs)
        return Hanc

    def perturbation(self):
        """Generation of the perturbation part of the gadget Hamiltonian 
        corresponding to the computational Hamiltonian of the Holmes2021 paper
        Args: None
        Returns:
            observable (qml.Hamiltonian)    : perturbation Hamiltonian
        """
        coeffs = [self.l] * self.n_anc
        obs = []
        for anc_qubit in range(self.n_anc):           # /!\ only valid for 2-local
            term = qml.PauliX(self.n_comp+anc_qubit)
            # TODO: substitute with and check
            # term = qml.operation.Tensor(term, *[qml.PauliZ((self.loc-1)*anc_qubit+q) for q in range(self.loc-1)])
            for q in range(self.loc-1):
                term = term @ qml.PauliZ((self.loc-1)*anc_qubit+q)
            obs += [term]
        V = qml.Hamiltonian(coeffs, obs)
        return V
    
    def gadget(self):
        """Generation of the perturbative gadget decomposition of the 
        computational Hamiltonian of the Holmes2021 paper to be used as a 
        cost function with qml.ExpvalCost
        Args: None
        Returns:
            observable (qml.Hamiltonian)    : gadget Hamiltonian to be used as observable
        """
        Hgad = self.ancillary() + self.perturbation()
        return Hgad

    def cat_projector(self):
        """Generation of a projector on the cat state (|00...0> + |11...1>)/sqrt(2)
        to be used as a cost function with qml.ExpvalCost
        Args: None
        Returns:
            observable (qml.Hamiltonian)    : projector
        """
        cat_state = np.zeros(2**self.n_anc)
        cat_state[[0, -1]] = 1/np.sqrt(2) 
        cat_projector = qml.Hermitian(np.outer(cat_state, cat_state), range(self.n_comp, self.n_tot, 1))
        return qml.Hamiltonian([1], [cat_projector])

    def ancillary_ground_projector(self):
        """Generation of a projector on the ground state of the ancillary 
        part of the gadget Hamiltonian: span{|00...0>, |11...1>}
        to be used as a cost function with qml.ExpvalCost
        Args: None
        Returns:
            observable (qml.Hamiltonian)    : projector
        """
        zeros_state = [0] * self.n_anc
        ones_state = [1] * self.n_anc
        obs = [qml.Projector(basis_state=zeros_state, wires=range(self.n_comp, self.n_tot, 1)), 
               qml.Projector(basis_state=ones_state, wires=range(self.n_comp, self.n_tot, 1))]
        coeffs = [1, 1]
        projector = qml.Hamiltonian(coeffs, obs)
        return projector
    
    def computational_ground_projector(self):
        """Generation of a projector on the ground state of the computational 
        Hamiltonian, e.g.: span{|001>, |010>, |100>, |111>}
        to be used as a cost function with qml.ExpvalCost
        Args: None
        Returns:
            observable (qml.Hamiltonian)    : projector
        """
        odd_numbers = np.arange(self.n_comp)
        odd_numbers = odd_numbers[odd_numbers % 2 == 1]
        obs = []
        for bit_flips in odd_numbers:
            # all combinations of bit_flips X gates on the n_comp qubits
            state = np.zeros(self.n_comp)
            state[0:bit_flips] = 1
            for perm in multiset_permutations(state):
                perm = [int(i) for i in perm]   # eliminating the tensor property
                obs += [qml.Projector(basis_state=perm, 
                                      wires=range(self.n_comp))]
        coeffs = np.ones(len(obs))
        projector = qml.Hamiltonian(coeffs, obs)
        return projector
