import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from hardware_efficient_ansatz import AlternatingLayeredAnsatz, SimplifiedAlternatingLayeredAnsatz

np.random.seed(42)
qml.drawer.use_style('black_white')


def test1():
    width = 4
    num_layers = 3
    dev = qml.device("default.qubit", wires=range(width))
    ala = AlternatingLayeredAnsatz()
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
    width = 4
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


if __name__ == "__main__":
    # test1()
    test2()
    # test3()