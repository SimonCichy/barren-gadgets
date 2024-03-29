import pennylane as qml
from pennylane import numpy as np


class AlternatingLayeredAnsatz:
    """
    Class to generate an ansatz circuit which can be used with qml.ExpvalCost.
    Args:
        gate_sequence (np.array[qml.operation]): array of single qubit gates to
                        be applied in the circuit. Should have dimentions
                        (number of layers, number of qubits)
        initial_y_rot (bool)                  : whether to apply a pi/4 Y-rotation
                                                to all qubits at the beginning
        coupling_pattern (string)             : what kind of CZ coupling gate
                                                patter to use. Should be "ladder"
                                                or "alternating".
    Returns:
        (object)
    """

    name = "Alternating Layered Ansatz"

    def __init__(self, gate_sequence=None, initial_y_rot=True, 
                #  cat_range=None, undo_cat=False, 
                 coupling_pattern = "ladder"
                 ):
        self.gate_sequence = gate_sequence
        self.do_y = initial_y_rot
        # self.cat_range = cat_range
        # self.undo_cat = undo_cat
        self.coupling_pattern = coupling_pattern
    
    def generate_sequence(self, n_qubits, n_layers):
        """Generation of a random sequence of Pauli gate rotations to be used in the Alternating Layered Ansatz

        Args:
            n_qubits (int)  : number of qubits (width) of the circuit to be filled with rotations
            n_layers (int)  : number of layers (depth) of the circuit to be filled with rotations

        Returns:
            self.gate_sequence (list)   : list of rotations to be implemented in each layer on each qubit.
                                          should be of dimension (n_layers, n_qubits)
        """
        gate_set = [qml.RX, qml.RY, qml.RZ]
        random_gate_sequence = [[np.random.choice(gate_set) for _ in range(n_qubits)] 
                                                            for _ in range(n_layers)]
        self.gate_sequence = random_gate_sequence
    
    # def cat_state_preparation(self, wires):
    #     """Generating a cat state on the specified qubits (e.g. the ancillary qubits for the perturbative gadget implementation)
    #     Resulting state: (|00...0> + |11...1>)/sqrt(2) 

    #     Args:
    #         wires (list)                : list of wires to apply the rotations on

    #     Returns:
    #         (callable) quantum function preparing the said state to be used in some other quantum function (e.g. self.ansatz)
    #     """
    #     # leave the computational qubits in the |0> state
    #     # create a cat state |+> in the ancillary register
    #     # /!\ inefficient in depth
    #     qml.Hadamard(wires=wires[0])
    #     for qubit in wires[1:]:
    #         qml.CNOT(wires=[wires[0], qubit])
    
    # def cat_state_undoing(self, wires):
    #     """Unduing a cat state on the specified qubits 
    #     (e.g. the ancillary qubits for the perturbative gadget implementation)

    #     Args:
    #         wires (list)                : list of wires to apply the rotations on

    #     Returns:
    #         (callable) quantum function preparing the said state to be used in some other quantum function (e.g. self.ansatz)
    #     """
    #     for qubit in wires[1:]:
    #         qml.CNOT(wires=[wires[0], qubit])
    #     qml.Hadamard(wires=wires[0])
    
    def ansatz(self, params, wires):
        """Generating the circuit corresponding to an Alternating Layerd Ansatz 
        (McClean2018, and Holmes2021)

        Args:
            params (array[float]) : array of parameters of dimension 
                                    (num_layers, num_qubits) containing the 
                                    rotation angles. Should have the same 
                                    dimentions as self.gate_sequence as they are 
                                    used pairwise
            wires (list)          : list of wires to apply the rotations on

        Returns:
            (callable) quantum function representing the ansatz 
                        (to be used e.g. with qml.ExpvalCost)
        """
        
        assert(len(np.shape(params)) == 2)      # check proper dimensions of params
        num_layers = np.shape(params)[0]        # np.shape(params) = (num_layers, num_qubits)
        num_qubits = np.shape(params)[1]
        assert(len(wires) == num_qubits)
        if self.gate_sequence == None:
            self.generate_sequence(num_qubits, num_layers)
        if self.do_y:
            for i in wires:
                qml.RY(np.pi/4, wires=i)
        # if self.cat_range is not None:
        #     self.cat_state_preparation(self.cat_range)
        for l in range(num_layers):
            # Single random gate layer (single qubit rotations)
            for i in wires:
                self.gate_sequence[l][i](params[l][i], wires=i)
            # Nearest neighbour controlled phase gates
            if num_qubits > 1:                          # no entangling gates if using a single qubit
                # qml.broadcast(qml.CZ, wires=wires, pattern="ring")
                if self.coupling_pattern == "ladder": 
                    qml.broadcast(qml.CZ, wires=wires, pattern="double")
                    qml.broadcast(qml.CZ, wires=wires, pattern="double_odd")
                if self.coupling_pattern == "alternating":
                    parity = l % 2
                    if parity == 0:                        # even layers
                        qml.broadcast(qml.CZ, wires=wires, pattern="double")
                    else:                                  # odd layers
                        qml.broadcast(qml.CZ, wires=wires, pattern="double_odd")
                        if num_qubits % 2 == 0:            # even number of qubits 
                            qml.CZ(wires = [0, num_qubits-1])
        # if self.cat_range is not None:
        #     if self.undo_cat:
        #         self.cat_state_undoing(self.cat_range)


class SimplifiedAlternatingLayeredAnsatz(AlternatingLayeredAnsatz):
    """Generating the circuit corresponding to an Alternating Layerd Ansatz 
        with only Y rotations and CZ coupling gates
        (Cerezo2021)

        Args:
            params (array[float]) : array of parameters of dimension 
                                    (num_layers, num_qubits) containing the 
                                    rotation angles. 
            wires (list)          : list of wires to apply the rotations on

        Returns:
            (callable) quantum function representing the ansatz 
                        (to be used e.g. with qml.ExpvalCost)
        """

    name = "Simplified Alternating Layered Ansatz"

    def __init__(self, width, depth, initial_y_rot=False
                #  , cat_range=None
                 ):
        super().__init__(initial_y_rot=initial_y_rot
                        #  , cat_range=cat_range
                         )
        self.gate_sequence = [[qml.RY for _ in range(width)] 
                                      for _ in range(depth)]
    
    def ansatz(self, params, wires):
        """Generating the circuit corresponding to a simplified 
        Alternating Layerd Ansatz (FIG.4. Cerezo2021)

        Args:
            params (array[float]) : array of parameters of dimension 
                                    (num_layers, num_qubits) containing the 
                                    rotation angles. 
            wires (list)          : list of wires to apply the rotations on

        Returns:
            (callable) quantum function representing the ansatz 
                        (to be used e.g. with qml.ExpvalCost)
        """
        
        assert(len(np.shape(params)) == 2)      # check proper dimensions of params
        num_layers = np.shape(params)[0]        # np.shape(params) = (num_layers, num_qubits)
        num_qubits = np.shape(params)[1]
        assert(len(wires) == num_qubits)
        if self.do_y:
            for i in wires:
                qml.RY(np.pi/4, wires=i)
        # if self.cat_range is not None:
        #     self.cat_state_preparation(self.cat_range)
        for l in range(num_layers):
            parity = l % 2
            # Single random gate layer (single qubit Y rotations)
            target_wires = wires[parity: parity+int(2*np.floor((num_qubits-parity)/2))] 
            for i in target_wires:
                self.gate_sequence[l][i](params[l][i], wires=i)
            # Nearest neighbour controlled phase gates
            if parity == 0:                          # even layers
                qml.broadcast(qml.CZ, wires=wires, pattern="double")
            else:                                   # odd layers
                qml.broadcast(qml.CZ, wires=wires, pattern="double_odd")


class HardwareEfficientAnsatz:
    """Deprecated: replaced by AlternatingLayeredAnsatz"""
    def __init__(self, gate_sequence=None, initial_y_rot=True, cat_range=None):
        self.gate_sequence = gate_sequence
        self.do_y = initial_y_rot
        self.cat_range = cat_range
    
    def generate_sequence(self, n_qubits, n_layers):
        """Generation of a random sequence of Pauli gate rotations to be used in the Alternating Layered Ansatz

        Args:
            n_qubits (int)  : number of qubits (width) of the circuit to be filled with rotations
            n_layers (int)  : number of layers (depth) of the circuit to be filled with rotations

        Returns:
            self.gate_sequence (list)   : list of rotations to be implemented in each layer on each qubit.
                                          should be of dimension (n_layers, n_qubits)
        """
        gate_set = [qml.RX, qml.RY, qml.RZ]
        random_gate_sequence = [[np.random.choice(gate_set) for _ in range(n_qubits)] for _ in range(n_layers)]
        self.gate_sequence = random_gate_sequence
    
    def cat_state_preparation(self, wires):
        """Generating a cat state on the specified qubits (e.g. the ancillary qubits for the perturbative gadget implementation)
        Resulting state: (|00...0> + |11...1>)/sqrt(2) 

        Args:
            wires (list)                : list of wires to apply the rotations on

        Returns:
            (callable) quantum function preparing the said state to be used in some other quantum function (e.g. self.ansatz)
        """
        # leave the computational qubits in the |0> state
        # create a cat state |+> in the ancillary register
        # /!\ inefficient in depth
        qml.Hadamard(wires=wires[0])
        for qubit in wires[1:]:
            qml.CNOT(wires=[wires[0], qubit])
    
    def ansatz(self, params, wires):
        """Generating the circuit corresponding to an Alternating Layerd Ansatz (McClean2018, Cerezo2021 and Holmes2021)

        Args:
            params (array[array[float]]): array of parameters of dimension (num_layers, num_qubits) containing the rotation angles
                                          should have the same dimentions as self.gate_sequence as they are used pairwise
            wires (list)                : list of wires to apply the rotations on

        Returns:
            (callable) quantum function representing the ansatz (to be used e.g. with qml.ExpvalCost)
        """
        
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

