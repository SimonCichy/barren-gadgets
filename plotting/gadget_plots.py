import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.lines import Line2D


def plot_variances_vs_qubits(file_list, colours, normalize=False, limits=None,  lambda_value = 1):
    fig, ax = plt.subplots()
    ax2 = ax.twiny() 
    xlim = [100, 0]

    r = 1
    # linestyles = ['--', ':', '-', '-.']

    for f, file in enumerate(file_list):
        data = np.loadtxt(file)
        layers = data[:,0].astype(int)
        qubits = data[:,1].astype(int)
        gradients = data[:,2:]
        xlim = [min([xlim[0], min(qubits)]), max([xlim[1], max(qubits)])]

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
            lam = lambda_value * (qubits_list - 1) / (4 * qubits_list)
            if 'global' in file: 
                norm = r if normalize else 1
                ax.semilogy(qubits_list, variance_vals/norm**2, "--s", c=colours[f][nl])
            elif 'local' in file: 
                norm = 1
                ax.semilogy(qubits_list, variance_vals/norm**2, ":v", c=colours[f][nl])
            elif 'gadget' in file:
                max_idx = len(qubits_list)
                if type(normalize) is dict:
                    norm = normalize['gadget']
                    max_idx = len(norm)
                elif normalize:
                    kprime = int(file[file.find('gadget') + 6])
                    ktilde = k / (kprime - 1)
                    norm = 0.5 * r * ktilde * (ktilde - 1) + r * lam * ktilde
                else: 
                    norm = 1
                ax2.semilogy(2*qubits_list[:max_idx], variance_vals[:max_idx]/norm**2, "-o", c=colours[f][nl])

    # ax.set_ylim([2.5e-3, 2.2e-1])
    ax.set_xlabel(r"N Computational Qubits")
    ax.set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel(r"N Total Qubits", color = 'grey') 
    ax2.tick_params(axis ='x', labelcolor = 'grey')
    ax2.xaxis.set_major_locator(MultipleLocator(2))

    xlim = [xlim[0]-0.5, xlim[1]+0.5]
    ax.set_xlim(xlim)
    ax2.set_xlim([2*xlim[0], 2*xlim[1]])
    if limits != None:
        ax.set_ylim(limits)

    # selection = 1 if np.shape(colours)[0] == 3 else 0
    selection = -1
    custom_lines = [Line2D([0], [0], color=colours[selection][nl], lw=2) for nl in range(len(layers_list))]
                #    [Line2D([0], [0], color='grey', linestyle=linestyles[c], lw=2) for c in range(len(file_list))]
    ax.legend(custom_lines, ['1 layer']+['{} layers'.format(num_layers) for num_layers in layers_list[1:]])
    # ax.legend(ncol=2)

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



def plot_training(file_list, colours, limits=None):
    fig, ax = plt.subplots()
    ax2 = ax.twinx() 
    legends = [r'$\langle \psi_{HE}| H^{comp} |\psi_{HE} \rangle$', 
               r'$\langle \psi_{HE}| H^{gad} |\psi_{HE} \rangle$',
               r'$\langle \psi_{HE}| H^{anc} |\psi_{HE} \rangle$',
               r'$\langle \psi_{HE}| \lambda V |\psi_{HE} \rangle$']

    for f, file in enumerate(file_list):
        data = np.loadtxt(file)
        iterations = data[:,0].astype(int)
        cost_values = data[:,1:]

        for observable in range(1, 2, 1):
            ax.plot(iterations, cost_values[:, observable], c=colours[f][observable], label=legends[observable])
        ax2.plot(iterations, cost_values[:, 0], c=colours[f][0], label=legends[0])
    
    ax.set_xlabel(r"Number of iterations")
    ax.set_ylabel(r"Gadget cost function")
    ax2.set_ylabel(r"Computational cost function")
    # ax.legend()

    if limits != None:
        ax.set_ylim(limits)

    plt.show()
