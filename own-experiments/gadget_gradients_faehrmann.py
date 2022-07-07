import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from faehrmann_gadgets import NewPerturbativeGadgets
from hardware_efficient_ansatz import AlternatingLayeredAnsatz

np.random.seed(42)

# General parameters:
num_samples = 200
layers_list = [1, 2, 5]         # [1, 2, 5, 10, 20, 50]
# layers_list = 'linear'
qubits_list = [2, 4, 6]               # [2, 3, 4, 5, 6]
lambda_scaling = 1                        # w.r.t. λ_max
gate_set = [qml.RX, qml.RY, qml.RZ]

gadgetizer = NewPerturbativeGadgets(perturbation_factor=lambda_scaling)

colours = np.array([plt.cm.Purples(np.linspace(0, 1, 10)),          # global
                    plt.cm.Blues(np.linspace(0, 1, 10)),            # local
                    plt.cm.Oranges(np.linspace(0, 1, 10)),          # gadget2
                    plt.cm.Reds(np.linspace(0, 1, 10)),             # gadget3
                    plt.cm.Greys(np.linspace(0, 1, 10))]            # legends
                    ).numpy()[:, 3:]
colours = colours[0]


if __name__ == "__main__":
    width_list = []
    norms_list = []
    variances_list = [[] for _ in range(len(layers_list))]
    for computational_qubits in qubits_list:
        term1 = qml.operation.Tensor(*[qml.PauliZ(q) for q in range(computational_qubits)])
        # term2 = qml.operation.Tensor(*[qml.PauliX(q) for q in range(computational_qubits)])
        Hcomp = qml.Hamiltonian([1], [term1])
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality=3)
        obs = Hcomp
        total_qubits, _, _ = gadgetizer.get_params(obs)
        print(total_qubits)
        width = total_qubits
        width_list += [width]
        norms_list += [np.sum(obs.coeffs)]
        for nl, num_layers in enumerate(layers_list):
            gradients_list = []
            for _ in range(num_samples):
                params = np.random.uniform(0, np.pi, size=(num_layers, width))
                random_gate_sequence = [[np.random.choice(gate_set) 
                                        for _ in range(width)] 
                                        for _ in range(num_layers)]
                dev = qml.device("default.qubit", wires=range(width))
                ala = AlternatingLayeredAnsatz(random_gate_sequence)
                ansatz = ala.ansatz
                cost = qml.ExpvalCost(ansatz, obs, dev)
                gradient = qml.grad(cost)(params)
                gradients_list += [gradient[0][0]]
            variances_list[nl] += [np.var(gradients_list)]
    
    fig, ax = plt.subplots()
    for line in range(len(variances_list)):
        normalized_variances = variances_list[line]/norms_list[line]**2
        ax.semilogy(qubits_list, normalized_variances, "--s", c=colours[line])
    ax.set_xlabel(r"N Total Qubits")
    ax.set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()







