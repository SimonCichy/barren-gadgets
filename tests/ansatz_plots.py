import sys
# sys.path.append('../src')
sys.path.append('src')
sys.path.append('Pennylane-tutorial')
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from hardware_efficient_ansatz import AlternatingLayeredAnsatz, SimplifiedAlternatingLayeredAnsatz
from layered_ansatz import RandomizedSimplifiedTwoDesign

np.random.seed(42)
qml.drawer.use_style('black_white')


def test1():
    width = 6
    num_layers = 5
    dev = qml.device("default.qubit", wires=range(width))
    ala = AlternatingLayeredAnsatz(coupling_pattern = "alternating")
    params = np.random.uniform(0, np.pi, size=(num_layers, width))
    @qml.qnode(dev)
    def circuit(params):
        ala.ansatz(params, range(width))
        return qml.probs(wires=range(width))
    # fig, ax = qml.draw_mpl(circuit, decimals=2)(params)
    # fig.show()

    drawer = qml.draw(circuit)
    print(drawer(params))

def test2():
    width = 5
    num_layers = 5
    dev = qml.device("default.qubit", wires=range(width))
    ala = SimplifiedAlternatingLayeredAnsatz(width, num_layers)
    params = np.random.uniform(0, np.pi, size=(num_layers, width))
    @qml.qnode(dev)
    def circuit(params):
        ala.ansatz(params, range(width))
        return qml.probs(wires=range(width))
    # fig, ax = qml.draw_mpl(circuit, decimals=2)(params)
    # fig.show()

    drawer = qml.draw(circuit)
    print(drawer(params))

def test3():
    width = 5
    depth = 4
    target_wires = range(width)
    dev = qml.device("default.qubit", wires=target_wires)
    # shapes = qml.SimplifiedTwoDesign.shape(n_layers=depth, n_wires=width)
    shapes = RandomizedSimplifiedTwoDesign.shape(n_layers=depth, n_wires=width)
    initial_layer_weights = [np.pi] * shapes[0][0] 
    weights = np.zeros(shapes[1])

    @qml.qnode(dev)
    def circuit(weights):
        # qml.SimplifiedTwoDesign.compute_decomposition(initial_layer_weights, weights, wires=target_wires)
        RandomizedSimplifiedTwoDesign.compute_decomposition(initial_layer_weights, weights, wires=target_wires)
        return qml.probs(wires=target_wires)

    drawer = qml.draw(circuit)
    print(drawer(weights))


if __name__ == "__main__":
    # test1()
    # test2()
    test3()