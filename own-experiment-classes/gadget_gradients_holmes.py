import sys
sys.path.append('../src')
sys.path.append('src')
import numpy as np
import matplotlib.pyplot as plt

from gradients_holmes import GradientsHolmes
from gadget_plots import plot_variances_vs_qubits

np.random.seed(42)

# General parameters:
num_samples = 200
layers_list = [1, 2, 5]         # [1, 2, 5, 10, 20, 50]
qubits_list = [2, 4]               # [2, 3, 4, 5, 6]
lambda_scaling = 0.5                        # w.r.t. λ_max

colours = np.array([plt.cm.Purples(np.linspace(0, 1, 10)), 
                    plt.cm.Blues(np.linspace(0, 1, 10)),
                    plt.cm.Oranges(np.linspace(0, 1, 10)),
                    plt.cm.Reds(np.linspace(0, 1, 10)),
                    plt.cm.Greys(np.linspace(0, 1, 10))])[:, 3:]

if __name__ == "__main__":
    gh = GradientsHolmes(qubits_list, layers_list, lambda_scaling, num_samples)
    print("Global circuit: ")
    gh.generate("global")
    print("Local circuit: ")
    gh.generate("local")
    print("2-local gadget circuit: ")
    gh.generate("gadget2")
    print("3-local gadget circuit: ")
    gh.generate("gadget3")





