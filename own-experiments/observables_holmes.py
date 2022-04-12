import pennylane as qml


class ObservablesHolmes:
    def __init__(self, computational_qubits, ancillary_qubits, perturbation_factor):
        self.n_comp = computational_qubits
        self.n_anc = ancillary_qubits
        self.n_tot = self.n_comp + self.n_anc
        self.factor = perturbation_factor
        self.l = perturbation_factor * (self.n_comp - 1) / (4 * self.n_comp)
    
    def computational(self):
        Hcomp = qml.PauliZ(0)
        for qubit in range(self.n_comp-1):
            Hcomp = Hcomp @ qml.PauliZ(qubit + 1)
        return Hcomp
    
    def ancillary(self):
        coeffs = []
        obs = []
        for first_qubit in range(self.n_comp, self.n_tot):
            for second_qubit in range(first_qubit+1, self.n_tot):
                coeffs += [0.5, -0.5]
                obs += [qml.Identity(first_qubit) @ qml.Identity(second_qubit), qml.PauliZ(first_qubit) @ qml.PauliZ(second_qubit)]
        Hanc = qml.Hamiltonian(coeffs, obs)
        return Hanc
    
    def perturbation(self):
        coeffs = [self.l] * self.n_comp
        obs = []
        for qubit in range(self.n_comp):           # /!\ only valid for 2-local
            obs += [qml.PauliZ(qubit) @ qml.PauliX(self.n_comp+qubit)]
        V = qml.Hamiltonian(coeffs, obs)
        return V
    
    def gadget(self):
        Hgad = self.ancillary() + self.perturbation()
        return Hgad
