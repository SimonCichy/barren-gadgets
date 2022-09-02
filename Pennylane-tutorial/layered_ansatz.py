import pennylane as qml
from pennylane import numpy as np
from pennylane import SimplifiedTwoDesign

class RandomizedSimplifiedTwoDesign(SimplifiedTwoDesign):
    """ Based on the SimplifiedTwoDesign template from pennylane
    https://docs.pennylane.ai/en/latest/code/api/pennylane.SimplifiedTwoDesign.html
    as proposed in `Cerezo et al. (2021) <https://doi.org/10.1038/s41467-021-21728-w>`_.
    but changin the Y-rotations for a random choice of {X, Y, Z}-rotations.
    """


    def __init__(self, initial_layer_weights, weights, wires, do_queue=True, id=None):
        super().__init__(initial_layer_weights, weights, wires=wires, do_queue=do_queue, id=id)


    @staticmethod
    def compute_decomposition(initial_layer_weights, weights, wires):

        n_layers = qml.math.shape(weights)[0]
        op_list = []

        # initial rotations
        for i in range(len(wires)):
            op_list.append(qml.RY(initial_layer_weights[i], wires=wires[i]))

        # generating the rotation sequence
        gate_set = [qml.RX, qml.RY, qml.RZ]
        random_gate_sequence = np.random.choice(gate_set, size=qml.math.shape(weights))

        # repeated layers
        for layer in range(n_layers):

            # even layer of entanglers
            even_wires = [wires[i : i + 2] for i in range(0, len(wires) - 1, 2)]
            for i, wire_pair in enumerate(even_wires):
                op_list.append(qml.CZ(wires=wire_pair))
                op_list.append(random_gate_sequence[layer, i, 0].item()(weights[layer, i, 0], wires=wire_pair[0]))
                op_list.append(random_gate_sequence[layer, i, 1].item()(weights[layer, i, 1], wires=wire_pair[1]))
                # op_list.append(qml.RX(weights[layer, i, 0], wires=wire_pair[0]))
                # op_list.append(qml.RX(weights[layer, i, 1], wires=wire_pair[1]))

            # odd layer of entanglers
            odd_wires = [wires[i : i + 2] for i in range(1, len(wires) - 1, 2)]
            for i, wire_pair in enumerate(odd_wires):
                op_list.append(qml.CZ(wires=wire_pair))
                op_list.append(random_gate_sequence[layer, len(wires) // 2 + i, 0].item()(weights[layer, len(wires) // 2 + i, 0], wires=wire_pair[0]))
                op_list.append(random_gate_sequence[layer, len(wires) // 2 + i, 1].item()(weights[layer, len(wires) // 2 + i, 1], wires=wire_pair[1]))
                # op_list.append(qml.RX(weights[layer, len(wires) // 2 + i, 0], wires=wire_pair[0]))
                # op_list.append(qml.RX(weights[layer, len(wires) // 2 + i, 1], wires=wire_pair[1]))

        return op_list

