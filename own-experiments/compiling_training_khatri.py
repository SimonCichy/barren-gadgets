import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
# import datetime

from compiling_khatri import CompilingKhatri

seed = 2
np.random.seed(seed)
# data_folder = '../results/data/training/'
plot_data = True
save_data = True

computational_qubits = 6
max_iter = 50
step = 3
print_frequency = 10
num_shots = 1000

dev = qml.device("default.qubit", 
                 wires=range(2*computational_qubits), 
                 shots=num_shots)
opt = qml.GradientDescentOptimizer(stepsize=step)
example = 2
cp = CompilingKhatri(computational_qubits, example, dev)


if __name__ == "__main__":
    print("Training the compiler on example {} using the global cost".format(cp.example))
    print("Width of the target unitary:", computational_qubits)
    print("Iterations:                 ", max_iter)
    initial_weights = np.random.uniform(0, np.pi, 
                        size=np.shape(cp.params_ex[cp.example]), 
                        requires_grad=True)
    cost_g_list = [cp.cost_global(initial_weights)]
    weights = initial_weights
    for it in range(max_iter):
        weights = opt.step(cp.cost_global, weights)
        cost_g_list.append(cp.cost_global(weights))
        if (it+1) % print_frequency == 0:
            print("Iteration = {:5d} | ".format(it+1) + 
                    "Cost function = {: .8f}".format(cost_g_list[-1]))
    print("Rotation angles to be learned: ")
    print(cp.params_ex[example])
    print("Learned angles: ")
    print(weights)

    print("Training the compiler on example {} using the local cost".format(cp.example))
    print("Width of the target unitary:", computational_qubits)
    print("Iterations:                 ", max_iter)
    cost_l_list = [cp.cost_local(initial_weights)]
    witness_g_list = [cp.cost_global(initial_weights)]
    weights = initial_weights
    for it in range(max_iter):
        weights = opt.step(cp.cost_local, weights)
        cost_l_list.append(cp.cost_local(weights))
        witness_g_list.append(cp.cost_global(weights))
        if (it+1) % print_frequency == 0:
            print("Iteration = {:5d} | ".format(it+1) + 
                    "Cost function = {: .8f}".format(cost_l_list[-1]))
    print("Rotation angles to be learned: ")
    print(cp.params_ex[example])
    print("Learned angles: ")
    print(weights)

    if plot_data:
        plt.figure()
        plt.plot(range(max_iter+1), cost_g_list, '-sb', label="HST")
        plt.plot(range(max_iter+1), cost_l_list, '-or', label="LHST")
        plt.plot(range(max_iter+1), witness_g_list, '-vg', label="HST via LHST")
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("{} qubits".format(computational_qubits))
        plt.legend()
    if save_data:
        pass
    plt.show()
