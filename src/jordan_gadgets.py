import pennylane as qml
from pennylane import numpy as np


class PerturbativeGadgets:
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
    def __init__(self, method='Jordan', perturbation_factor=1):
        self.method = method
        self.perturbation_factor = perturbation_factor
    
    def gadgetize(self, Hamiltonian, target_locality=2):
        """Generation of the perturbative gadget equivalent of the given 
        Hamiltonian according to the proceedure in Jordan2012
        Args:
            Hamiltonian (qml.Hamiltonian)   : target Hamiltonian to decompose
                                              into more local terms
            target_locality (int)           : locality of the resulting 
                                              gadget Hamiltonian
        Returns:
            Hgad (qml.Hamiltonian)          : gadget Hamiltonian
        """
        # checking for unaccounted for situations
        self.run_checks(Hamiltonian)
        # checking how many qubits the Hamiltonian acts on
        computational_qubits = len(Hamiltonian.wires)
        # total qubit count, updated progressively when adding ancillaries
        total_qubits = computational_qubits
        # getting the locality, assuming all terms have the same
        computational_locality = len(Hamiltonian.ops[0].non_identity_obs)
        l = self.perturbation_factor * (computational_locality - 1) / (4 * computational_locality)
        # creating the gadget Hamiltonian
        coeffs_anc = []
        coeffs_pert = []
        obs_anc = []
        obs_pert = []
        for str_count, string in enumerate(Hamiltonian.ops):
            ancillary_qubits = int(len(string.non_identity_obs) / (target_locality - 1))
            previous_total = total_qubits
            total_qubits += ancillary_qubits
            # Generating the ancillary part
            for first in range(previous_total, total_qubits):
                for second in range(first+1, total_qubits):
                    coeffs_anc += [0.5, -0.5]
                    obs_anc += [qml.Identity(first) @ qml.Identity(second), 
                            qml.PauliZ(first) @ qml.PauliZ(second)]
            # Generating the perturbative part
            for anc_q in range(ancillary_qubits):
                term = qml.PauliX(previous_total+anc_q)
                term = qml.operation.Tensor(term, *string.non_identity_obs[
                        (target_locality-1)*anc_q:(target_locality-1)*(anc_q+1)])
                obs_pert.append(term)
            coeffs_pert += [l * Hamiltonian.coeffs[str_count]] + [l] * (ancillary_qubits - 1)
        Hanc = qml.Hamiltonian(coeffs_anc, obs_anc)
        Hpert = qml.Hamiltonian(coeffs_pert, obs_pert)
        return Hanc + Hpert
    
    def run_checks(self, Hamiltonian):
        computational_qubits = len(Hamiltonian.wires)
        if computational_qubits != Hamiltonian.wires[-1] + 1:
            raise Exception('The studied computational Hamiltonian is not acting on ' + 
                            'the first {} qubits. '.format(computational_qubits) + 
                            'Decomposition not implemented for this case')
        # Check for same string lengths
        localities=[]
        for string in Hamiltonian.ops:
            localities.append(len(string.non_identity_obs))
        if len(np.unique(localities)) > 1:
            raise Exception('The given Hamiltonian has terms with different locality.' +
                            ' Gadgetization not implemented for this case')

    def eliminate_minus_subspace(Hamiltonian):
        """Method to discard the block of the Hamiltonian acting on the -1 
        subspace of the X^n operator"""
        pass


