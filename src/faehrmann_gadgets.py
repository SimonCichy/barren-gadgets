import pennylane as qml
from pennylane import numpy as np
import copy


class NewPerturbativeGadgets:
    """ Class to generate the gadget Hamiltonian corresponding to a given
    computational hamiltonian according to the gadget construction derived
    by Faehrmann & Cichy 
    
    Args: 
        perturbation_factor (float) : parameter controlling the magnitude of the
                                      perturbation (aa pre-factor to \lambda_max)
    """
    def __init__(self, perturbation_factor=1):
        #TODO: move perturbation factor to gadgetize and eliminate class?
        self.perturbation_factor = perturbation_factor
    
    def map_wires(self, H, wires_map):
        """Map the wires of an Observable according to a wires map.
        
        Args:
            H (Hamiltonian or Tensor or Observable): Hamiltonian to remap the wires of.
            wires_map (dict): Wires map with `(origin, destination)` pairs as key-value pairs.
        
        Returns:
            Hamiltonian or Tensor or Observable: A copy of the original Hamiltonian with remapped wires.
        """
        if isinstance(H, qml.Hamiltonian):
            new_ops = [self.map_wires(op, wires_map) for op in H.ops]
            new_H = qml.Hamiltonian(H.coeffs, new_ops)
        elif isinstance(H, qml.operation.Tensor):
            new_obs = [self.map_wires(ob, wires_map) for ob in H.obs]
            new_H = qml.operation.Tensor(*new_obs)
        elif isinstance(H, qml.operation.Observable):
            new_H = copy.copy(H)
            new_H._wires = new_H.wires.map(wires_map)
            
        return new_H
    
    def reorder_qubits(self, Hcomp, Hgad, ordering="rotating"):
        """Generating a new Hamiltonian object corresponding to the gadgetiized
        Hamiltonian but changing the order of the different qubits in the 
        register to help the optimization
        Args:
            Hcomp (qml.Hamiltonian)   : original gadgetized Hamiltonian 
            Hgad (qml.Hamiltonian)    : gadgetized Hamiltonian to be 
                                        reordered
            ordering (str)            : how to order the qubits from the 
                                        different registers. Should be 
                                        "comp-aux" or "rotating"
        Returns:
            Hgad (qml.Hamiltonian)    : gadget Hamiltonian
        """
        if ordering == "comp-aux":
            new_Hgad = Hgad
        elif ordering == "rotating":
            # extracting all relevant parameters from the two Hamiltonians
            computational_qubits, computational_locality, computational_terms = self.get_params(Hcomp)
            total_qubits, target_locality, _ = self.get_params(Hcomp)
            # more compact notation
            n_comp = computational_qubits
            k = computational_locality
            r = computational_terms
            n_tot = total_qubits
            k_prime = target_locality
            # Starting with the n_comp computational qubits
            new_order = np.arange(n_comp)
            for string in Hgad.ops:
                if type(string) is not qml.Identity:
                    # print(string)
                    affected_qubits = string.wires.toarray()
                    # print(affected_qubits)
                    if len(affected_qubits) > 1:
                        computational_target = int(min(affected_qubits))
                        # print(computational_target)
                        auxiliary_qubits = np.setdiff1d(affected_qubits, computational_target)
                        # print(auxiliary_qubits)
                        added = False
                        target_index = np.where(new_order == computational_target)[0] + 1
                        while added is False:
                            if min(auxiliary_qubits) in new_order:
                                auxiliary_qubits = np.delete(auxiliary_qubits, np.argmin(auxiliary_qubits))
                            else:
                                new_order = np.insert(new_order, target_index, min(auxiliary_qubits))
                                added = True
            print(new_order)
            # Generating the new Hamiltonian
            wires_map = {}
            for i in range(len(new_order)):
                wires_map[int(new_order[i])] = i
            new_Hgad = self.map_wires(Hgad, wires_map)
        else:
            print("Requested reordering scheme not implemented."
                  "Returning the Hamiltonian unchanged")
        return new_Hgad

    
    def gadgetize(self, Hamiltonian, target_locality=3):
        """Generation of the perturbative gadget equivalent of the given 
        Hamiltonian according to the proceedure in Cichy, Fährmann et al.
        Args:
            Hamiltonian (qml.Hamiltonian) : target Hamiltonian to decompose
                                            into more local terms
            target_locality (int > 2)     : desired locality of the resulting 
                                            gadget Hamiltonian
        Returns:
            Hgad (qml.Hamiltonian)          : gadget Hamiltonian
        """
        # checking for unaccounted for situations
        self.run_checks(Hamiltonian, target_locality)
        computational_qubits, computational_locality, computational_terms = self.get_params(Hamiltonian)
        
        # total qubit count, updated progressively when adding ancillaries
        total_qubits = computational_qubits
        #TODO: check proper convergence guarantee
        gap = 1
        perturbation_norm = np.sum(np.abs(Hamiltonian.coeffs)) \
                          + computational_terms * (computational_locality - 1)
        lambda_max = gap / (4 * perturbation_norm)
        l = self.perturbation_factor * lambda_max
        sign_correction = (-1)**(computational_locality % 2 + 1)
        # creating the gadget Hamiltonian
        coeffs_anc = []
        coeffs_pert = []
        obs_anc = []
        obs_pert = []
        ancillary_register_size = int(computational_locality / (target_locality - 2))
        for str_count, string in enumerate(Hamiltonian.ops):
            previous_total = total_qubits
            total_qubits += ancillary_register_size
            # Generating the ancillary part
            for anc_q in range(previous_total, total_qubits):
                coeffs_anc += [0.5, -0.5]
                obs_anc += [qml.Identity(anc_q), qml.PauliZ(anc_q)]
            # Generating the perturbative part
            for anc_q in range(ancillary_register_size):
                term = qml.PauliX(previous_total+anc_q) @ qml.PauliX(previous_total+(anc_q+1)%ancillary_register_size)
                term = qml.operation.Tensor(term, *string.non_identity_obs[
                    (target_locality-2)*anc_q:(target_locality-2)*(anc_q+1)])
                obs_pert.append(term)
            coeffs_pert += [l * sign_correction * Hamiltonian.coeffs[str_count]] \
                        + [l] * (ancillary_register_size - 1)
        coeffs = coeffs_anc + coeffs_pert
        obs = obs_anc + obs_pert
        Hgad = qml.Hamiltonian(coeffs, obs)
        return Hgad

    def get_params(self, Hamiltonian):
        """ retrieving the parameters n, k and r from the given Hamiltonian
        Args:
            Hamiltonian (qml.Hamiltonian) : Hamiltonian from which to get the
                                            relevant parameters
        Returns:
            computational_qubits (int)    : total number of qubits acted upon by 
                                            the Hamiltonian
            computational_locality (int)  : maximum number of qubits acted upon
                                            by a single term of the Hamiltonian
            computational_terms (int)     : number of terms in the sum 
                                            composing the Hamiltonian
        """
        # checking how many qubits the Hamiltonian acts on
        computational_qubits = len(Hamiltonian.wires)
        # getting the number of terms in the Hamiltonian
        computational_terms = len(Hamiltonian.ops)
        # getting the locality
        computational_locality = max([len(Hamiltonian.ops[s].non_identity_obs) 
                                      for s in range(computational_terms)])
        return computational_qubits, computational_locality, computational_terms
    
    def run_checks(self, Hamiltonian, target_locality):
        """ method to check a few conditions for the correct application of 
        the methods
        Args:
            Hamiltonian (qml.Hamiltonian) : Hamiltonian of interest
            target_locality (int > 2)     : desired locality of the resulting 
                                            gadget Hamiltonian
        Returns:
            None
        """
        computational_qubits, computational_locality, _ = self.get_params(Hamiltonian)
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
        # validity of the target locality given the computational locality
        if target_locality < 3:
            raise Exception('The target locality can not be smaller than 3')
        ancillary_register_size = computational_locality / (target_locality - 2)
        if int(ancillary_register_size) != ancillary_register_size:
            raise Exception('The locality of the Hamiltonian and the target' + 
                             ' locality are not compatible. The gadgetization' + 
                             ' with "unfull" ancillary registers is not' + 
                             ' supported yet. Please choose such that the' + 
                             ' computational locality is divisible by the' + 
                             ' target locality - 2')

    def zero_projector(self, Hamiltonian, target_locality=3):
        """Generation of a projector on the zero state |00...0>
        as a sum of projectors |0><0| on each qubit
        to be used as a cost function with qml.ExpvalCost
        Args: 
            Hamiltonian (qml.Hamiltonian) : Hamiltonian to be gadgetized
        Returns:
            projector (qml.Hamiltonian)   : projector that can be used as 
                                            an observable to measure
        """
        n_comp, k, r = self.get_params(Hamiltonian)
        ktilde = int(k/(target_locality-2))
        ancillary_qubits = r * ktilde
        coeffs = [1/ancillary_qubits] * ancillary_qubits
        obs = []
        for qubit in range(n_comp, n_comp+ancillary_qubits):
            zero_state = np.array([1, 0])
            zero_projector = qml.Hermitian(np.outer(zero_state, zero_state), 
                                           qubit)
            obs.append(zero_projector)
        projector = qml.Hamiltonian(coeffs, obs)
        return projector
    
    def all_zero_projector(self, Hamiltonian, target_locality=3):
        """Generation of a rank 1 projector on the zero state |00...0>
        to be used as a cost function with qml.ExpvalCost
        Args: 
            Hamiltonian (qml.Hamiltonian) : Hamiltonian to be gadgetized
        Returns:
            projector (qml.Hamiltonian)   : projector that can be used as 
                                            an observable to measure
        """
        n_comp, k, r = self.get_params(Hamiltonian)
        ktilde = int(k/(target_locality-2))
        ancillary_qubits = r * ktilde
        zero_state = np.zeros((2**ancillary_qubits))
        zero_state[0] = 1
        zero_projector = qml.Hermitian(np.outer(zero_state, zero_state), 
                                       range(n_comp, n_comp + ancillary_qubits, 1))
        projector = qml.Hamiltonian([1], [zero_projector])
        return projector


