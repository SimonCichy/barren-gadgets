import sys
sys.path.append('../src')
sys.path.append('src')
import time
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from faehrmann_gadgets import NewPerturbativeGadgets
from hardware_efficient_ansatz import AlternatingLayeredAnsatz, SimplifiedAlternatingLayeredAnsatz
from data_management import save_gradients

# General parameters:
num_samples = 1000
layers_list = [2, 5, 10, 20]
# layers_list = 'linear'
qubits_list = [4, 5, 6, 7, 8, 9]
# qubits_list = [4, 6, 8, 10, 12]
lambda_scaling = 1                        # w.r.t. lambda_max
gate_set = [qml.RX, qml.RY, qml.RZ]
newk = 3
# newk = 4
seed = 43

np.random.seed(seed)

gadgetizer = NewPerturbativeGadgets(perturbation_factor=lambda_scaling)

colours = np.array([plt.cm.Purples(np.linspace(0, 1, 10)),          # global
                    plt.cm.Blues(np.linspace(0, 1, 10)),            # local
                    plt.cm.Oranges(np.linspace(0, 1, 10)),          # gadget2
                    plt.cm.Reds(np.linspace(0, 1, 10)),             # gadget3
                    plt.cm.Greys(np.linspace(0, 1, 10))]            # legends
                    ).numpy()[:, 3:]
colours = colours[3]


if __name__ == "__main__":
    widths_list = []
    norms_list = []
    variances_list = [[] for _ in range(len(layers_list))]
    # gradients_lists_list = [[] for _ in range(len(layers_list))]
    gradients_all = {}
    runtimes_list = []
    data_dict = {
        'computational qubits': qubits_list,
        'layers': layers_list,
        'widths': widths_list,
        'norms': norms_list,
        'variances': variances_list, 
        # 'gradients': gradients_lists_list, 
        'all gradients': gradients_all,
        'runtimes': runtimes_list
    }
    filename = save_gradients(data_dict, perturbation_factor=lambda_scaling, 
                              random_seed=seed, mode='new file')

    for computational_qubits in qubits_list:
        tic = time.perf_counter()
        term1 = qml.operation.Tensor(*[qml.PauliZ(q) for q in range(computational_qubits)])
        Hcomp = qml.Hamiltonian([1], [term1])
        # obs = Hcomp
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality=newk)
        # obs = Hgad
        obs = Hcomp
        total_qubits = len(obs.wires)
        print('Computational qubits:          ', computational_qubits)
        print('Total qubits:                  ', total_qubits)
        width = total_qubits
        widths_list += [width]
        norms_list += [np.sum(np.abs(obs.coeffs))]
        save_gradients(obs=obs, mode='overwrite', filename=filename)
        for nl, num_layers in enumerate(layers_list):
            gradients_list = []
            for _ in range(num_samples):
                params = np.random.uniform(0, np.pi, size=(num_layers, width))
                random_gate_sequence = [[np.random.choice(gate_set) 
                                        for _ in range(width)] 
                                        for _ in range(num_layers)]
                dev = qml.device("default.qubit", wires=range(width))
                # ala = AlternatingLayeredAnsatz(random_gate_sequence)
                ala = SimplifiedAlternatingLayeredAnsatz(width, num_layers)
                ansatz = ala.ansatz
                cost = qml.ExpvalCost(ansatz, obs, dev)
                gradient = qml.grad(cost)(params)
                gradients_list += [gradient]
            # gradients_lists_list[nl] += [gradients_list]
            gradients_all[(computational_qubits, num_layers)] = gradients_list
            variances_list[nl] += [np.var(np.array(gradients_list)[:, 0, 0])]
            toc = time.perf_counter()
            print_statement = '{:<4d} layers,       runtime: {:11.0f} seconds'.format(num_layers, toc-tic)
            print(print_statement)
            save_gradients(update=print_statement, mode='overwrite', filename=filename)

        toc = time.perf_counter()
        runtimes_list += [toc-tic]
        print_statement = '{:<4d} total qubits, runtime: {:11.0f} seconds'.format(total_qubits, toc-tic)
        print(print_statement)
        data_dict = {
            'computational qubits': qubits_list,
            'layers': layers_list,
            'widths': widths_list,
            'norms': norms_list,
            'variances': variances_list, 
            # 'gradients': gradients_lists_list, 
            'all gradients': gradients_all,
            'runtimes': runtimes_list
        }
        save_gradients(data_dict, mode='overwrite', 
                       update=print_statement, filename=filename)
            
    fig, ax = plt.subplots()
    for line in range(len(variances_list)):
        normalized_variances = variances_list[line]/norms_list[line]**2
        ax.semilogy(qubits_list, normalized_variances, "--s", c=colours[line])
    ax.set_xlabel(r"N Total Qubits")
    ax.set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.show()







