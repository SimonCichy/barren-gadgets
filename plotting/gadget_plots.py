import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.lines import Line2D


def plot_variances_vs_qubits(file_list, colours, normalize=False):
    fig, ax = plt.subplots()
    ax2 = ax.twiny() 

    r = 1
    lam = 1

    for f, file in enumerate(file_list):
        data = np.loadtxt(file)
        layers = data[:,0].astype(int)
        qubits = data[:,1].astype(int)
        gradients = data[:,2:]

        # getting the different numbers of layers tested
        layers_list = np.unique(layers)
        
        for nl, num_layers in enumerate(layers_list):
            # getting the indexes of the corresponding rows
            row_indexes = np.where(layers==num_layers)
            # selecting the corresponding rows of the data
            qubits_list = qubits[row_indexes]
            grad_vals = gradients[row_indexes]
            variance_vals = np.var(grad_vals, axis=1)
            # rescaling factor
            k = qubits_list
            if 'global' in file: 
                norm = r if normalize else 1
                ax.semilogy(qubits_list, variance_vals/norm**2, "--s", c=colours[f][nl])
            elif 'local' in file: 
                norm = 1 if normalize else 1
                ax.semilogy(qubits_list, variance_vals/norm**2, ":v", c=colours[f][nl])
            elif 'gadget' in file:
                norm = r * k*(k-1) + lam * (1 + k - 1) if normalize else 1
                ax2.semilogy(2*qubits_list, variance_vals/norm**2, "-o", c=colours[f][nl])

    # ax.set_ylim([2.5e-3, 2.2e-1])
    ax.set_xlabel(r"N Computational Qubits")
    ax.set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel(r"N Total Qubits", color = 'grey') 
    ax2.tick_params(axis ='x', labelcolor = 'grey')
    ax2.xaxis.set_major_locator(MultipleLocator(2))

    selection = 1 if np.shape(colours)[0] == 3 else 0
    custom_lines = [Line2D([0], [0], color=colours[selection][nl], lw=2) for nl in range(len(layers_list))]
    ax.legend(custom_lines, ['1 layer']+['{} layers'.format(num_layers) for num_layers in layers_list[1:]])

    plt.show()

def plot_variances_vs_layers(file_list, colours, normalize=False):
    fig, ax = plt.subplots()
    ax2 = ax.twiny() 

    for f, file in enumerate(file_list):
        data = np.loadtxt(file)
        layers = data[:,0].astype(int)
        lamb = data[:,1]
        gradients = data[:,2:]

        # getting the different numbers of layers tested
        lambdas_list = np.unique(lamb)

        for l, lam in enumerate(lambdas_list):
            # getting the indexes of the corresponding rows
            row_indexes = np.where(lamb==lam)
            # selecting the corresponding rows of the data
            layers_list = layers[row_indexes]
            grad_vals = gradients[row_indexes]
            variance_vals = np.var(grad_vals, axis=1)
            # rescaling factor
            r = 1
            n = int(file[file.find('qubits')-1])
            k = n
            norm = r * k*(k-1) + lam * (1 + k - 1) if normalize else 1

            ax.semilogy(layers_list, variance_vals/norm**2, "-v", c=colours[f][l], label=r"$\lambda=${}".format(lam))

        ax.set_xlabel(r"Number of layers")
        ax.set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()
        ax.set_title(r"$n_{comp}=$"+"{}".format(n))

        plt.show()
