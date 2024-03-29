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
    
    def gadgetize(self, Hamiltonian, target_locality=2, 
                  offset_energy=False, penalization=None):
        """Generation of the perturbative gadget equivalent of the given 
        Hamiltonian according to the proceedure in Jordan2012
        Args:
            Hamiltonian (qml.Hamiltonian)   : target Hamiltonian to decompose
                                              into more local terms
            target_locality (int)           : locality of the resulting 
                                              gadget Hamiltonian
            offset_energy (bool)            : whether to shift all Hcomp to have
                                              only negative eigenenergies
            penalization (float)            : "regularization" constant to 
                                              penalize being away from |GHZ+>
        Returns:
            Hgad (qml.Hamiltonian)          : gadget Hamiltonian
        """
        computational_qubits, computational_locality, _ = self.get_params(Hamiltonian)
        # total qubit count, updated progressively when adding ancillaries
        total_qubits = computational_qubits
        #TODO: correct lambda_max for the case of more than 1 term and/or c_s
        l = self.perturbation_factor * (computational_locality - 1) / (4 * computational_locality)
        sign_correction = (-1)**(computational_locality % 2 + 1)
        # creating the gadget Hamiltonian
        coeffs_anc = []
        coeffs_pert = []
        obs_anc = []
        obs_pert = []
        ancillary_qubits = int(computational_locality / (target_locality - 1))
        for str_count, string in enumerate(Hamiltonian.ops):
            # ancillary_qubits = int(len(string.non_identity_obs) / (target_locality - 1))
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
                if (anc_q+1)*(target_locality-1) <= len(string.non_identity_obs):
                    # /!\ if the switch from Pauli to identity happens mid-register
                    #TODO: correct that 
                    # if anc_q*(target_locality-1) < len(string.non_identity_obs) < (anc_q+1)*(target_locality-1)
                    # term = qml.operation.Tensor(term, *string.non_identity_obs[
                        # (target_locality-1)*anc_q:])
                    term = qml.operation.Tensor(term, *string.non_identity_obs[
                        (target_locality-1)*anc_q:(target_locality-1)*(anc_q+1)])
                # else: 
                    # apply identities on some computational qubits -> superfluous
                obs_pert.append(term)
            coeffs_pert += [l * sign_correction * Hamiltonian.coeffs[str_count]] \
                        + [l] * (ancillary_qubits - 1)
        if offset_energy:
            previous_total = total_qubits
            total_qubits += ancillary_qubits
            # Generating the ancillary part, same as the others
            for first in range(previous_total, total_qubits):
                for second in range(first+1, total_qubits):
                    coeffs_anc += [0.5, -0.5]
                    obs_anc += [qml.Identity(first) @ qml.Identity(second), 
                            qml.PauliZ(first) @ qml.PauliZ(second)]
            # Generating the perturbative part, 
            # only identities on the computational register
            for anc_q in range(ancillary_qubits):
                term = qml.PauliX(previous_total+anc_q)
                # Is that even necessary?
                # term = qml.operation.Tensor(term, qml.Identity(wires=
                #        range(computational_qubits+anc_q*(target_locality-1), 
                #              computational_qubits+(anc_q+1)*(target_locality-1))))
                obs_pert.append(term)
            M = np.sum(Hamiltonian.coeffs)
            coeffs_pert += [l * sign_correction * M] + [l] * (ancillary_qubits - 1)
        coeffs = coeffs_anc + coeffs_pert
        obs = obs_anc + obs_pert
        if penalization is not None:
            P = self.cat_projector(Hamiltonian, target_locality)
            coeffs += [penalization*c for c in P.coeffs]
            obs += P.ops
        Hgad = qml.Hamiltonian(coeffs, obs)
        return Hgad

    def get_params(self, Hamiltonian):
        # checking for unaccounted for situations
        self.run_checks(Hamiltonian)
        # checking how many qubits the Hamiltonian acts on
        computational_qubits = len(Hamiltonian.wires)
        # getting the number of terms in the Hamiltonian
        computational_terms = len(Hamiltonian.ops)
        # getting the locality, assuming all terms have the same
        computational_locality = max([len(Hamiltonian.ops[s].non_identity_obs) 
                                      for s in range(computational_terms)])
        return computational_qubits, computational_locality, computational_terms
    
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

    def cat_projector(self, Hamiltonian, target_locality=2):
        """Generation of a projector on the cat state (|00...0> + |11...1>)/sqrt(2)
        to be used as a cost function with qml.ExpvalCost
        Args: 
            Hamiltonian (qml.Hamiltonian)   : Hamiltonian to be gadgetized
        Returns:
            observable (qml.Hamiltonian)    : projector
        /!\ |GHZ->^2 also has +1 eigenvalue
        """
        n_comp, k, r = self.get_params(Hamiltonian)
        ktilde = int(k/(target_locality-1))
        coeffs = [1] * r
        obs = []
        for register in range(r):
            target_qubits = range(n_comp + register * ktilde, 
                                  n_comp + (register + 1) * ktilde, 1)
            cat_state = np.zeros(2**ktilde)
            cat_state[[0, -1]] = 1/np.sqrt(2) 
            cat_projector = qml.Hermitian(np.outer(cat_state, cat_state), 
                                          target_qubits)
            obs.append(cat_projector)
        projector = qml.Hamiltonian(coeffs, obs)
        return projector
    
    def ancillary_X(self, Hamiltonian, target_locality=2):
        """Generation of the all X operator on the ancillary register. Its 
        expectation value should be close to 1 when the shifted cmputational
        Hamiltonian is minimized
        to be used as a cost function with qml.ExpvalCost
        Args: 
            Hamiltonian (qml.Hamiltonian)   : Hamiltonian to be gadgetized
        Returns:
            observable (qml.Hamiltonian)    : X^{\otimes r·target_k}
        """
        n_comp, k, r = self.get_params(Hamiltonian)
        ktilde = int(k/(target_locality-1))
        target_qubits = range(n_comp, n_comp + r * ktilde, 1)
        all_X = qml.operation.Tensor(*[qml.PauliX(q) for q in target_qubits])
        return qml.Hamiltonian([1], [all_X])

