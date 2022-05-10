import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import datetime

from compiling_khatri import CompilingKhatri

seed = 2
np.random.seed(seed)
# data_folder = '../results/data/training/'
plot_data = True
save_data = True

computational_qubits = 4
max_iter = 2000
step = 0.3
print_frequency = 100
num_shots = 1000

dev = qml.device("default.qubit", 
                 wires=range(2*computational_qubits), 
                 shots=num_shots)
opt = qml.GradientDescentOptimizer(stepsize=step)


if __name__ == "__main__":
    cp = CompilingKhatri(computational_qubits, 1)
    cost = cp.cost_global
    print("Training the compiler on example {self.example}")
    print("Width of the target unitary:", computational_qubits)
    print("Iterations:                 ", max_iter)
    initial_weights = np.random.uniform(0, np.pi, 
                        size=np.shape(cp.params_ex[cp.example]), 
                        requires_grad=True)
    cost_list = [cost(initial_weights)]
    weights = initial_weights
    for it in range(max_iter):
        weights = opt.step(cost, weights)
        cost_list.append(cost(weights))
        if (it+1) % print_frequency == 0:
            print("Iteration = {:5d} | ".format(it+1) + 
                    "Cost function = {: .8f}".format(cost_list[-1]))
    if plot_data:
        pass
    if save_data:
        pass
    plt.show()
