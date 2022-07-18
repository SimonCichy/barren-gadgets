import sys
sys.path.append('../src')
sys.path.append('src')
import matplotlib.pyplot as plt
import numpy as np
from gadget_plots import *
from data_management import get_training_costs, get_training_labels2
from rsmf import CustomFormatter

data_folder = '../results/data/'

# palette = ['#000000', '#252525', '#676767', '#ffffff', 
#            '#171723', '#004949', '#009999', '#22cf22', 
#            '#490092', '#006ddb', '#b66dff', '#ff6db6', 
#            '#920000', '#8f4e00', '#db6d00', '#ffdf4d']
palette = [['#e65e71', '#d64055', '#ba182e', '#851919'], 
           ['#77f2d8', '#64d4bc', '#3cb89d', '#279680'], 
           ['#00a1d9', '#1977c2', '#2041ba', '#202482'], 
           ['#b0b0b0', '#727272', '#606060', '#393939']]


abstract_formatter = CustomFormatter(
    columnwidth=246.0 * 0.01389,
    wide_columnwidth=510.0 * 0.01389,
    fontsizes=11,
    pgf_preamble=r"\usepackage[T1]{fontenc}\usepackage{dsfont}",
)

def training_plots(): 
    lambdas = [0.1, 1, 10]
    # file_list = ['220708_euler/training_nr00{}'.format(count) for count in [18, 20, 22]]
    file_list = ['220713_qmio/training_nr{:0>4}'.format(count) for count in [5, 7, 9]]
    palette_choice = [palette[0][2], palette[2][0]]

    fig = abstract_formatter.figure(aspect_ratio=.4, wide=True)
    axs = fig.subplots(1, 3)
    for f, file in enumerate(file_list):
        file = data_folder + 'training/' + file
        costs = get_training_costs(file + '.npz')
        labels = get_training_labels2(file + '.txt')
        axs[f].plot(np.ones(len(costs[0])), ':', c='gainsboro')
        axs[f].plot(-np.ones(len(costs[0])), ':', c='gainsboro')
        axs[f].plot(np.zeros(len(costs[0])), ':', c='gainsboro')
        for i in range(1, 3):
            axs[f].plot(costs[i],'-', c=palette_choice[i-1], label=labels[i])
        axs[f].set_title(r'$\lambda = {}$'.format(lambdas[f]) + r'$\lambda_{max}$')
    axs[0].set_ylabel('Cost')
    axs[1].set_xlabel('Number of iterations')
    custom_lines = [Line2D([0], [0], color=palette_choice[nl], lw=2) for nl in range(2)]
    lgd = fig.legend(handles=custom_lines,
                    labels=[r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| (H^{comp}\otimes\mathds{1}^{anc})]$', 
                            r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| H^{gad}]$'], 
                    loc='lower center',
                    bbox_to_anchor=(0.1, -0.12, 0.83, 0.5),
                    mode='expand',
                    ncol=4)
    plt.tight_layout()
    plt.savefig(data_folder + '../plots/training_new_gadget/trainings_for_abstract.pdf', 
                bbox_extra_artists = (lgd,), bbox_inches='tight')

def get_vars_for_plot(file, max_qubit=np.inf):
    data = np.load(file, allow_pickle=True)
    qubits_list = data['computational_qubits'][:len(data['widths_list'])]
    layers_list = data['layers_list']
    norms_list = data['norms_list']
    gradients_dict = data['all_gradients'].item()
    variances_list=[[] for _ in range(len(layers_list))]
    cutoff_index = np.where(qubits_list == max_qubit)
    if np.shape(cutoff_index[0]) == (1,):
        widths_range = qubits_list[:cutoff_index[0][0]+1]
    else: 
        widths_range = qubits_list
    for width in widths_range:
        for nl, depth in enumerate(layers_list):
            full_gradient = np.array(gradients_dict[(width, depth)])
            param_count = np.prod(np.shape(full_gradient[1:]))
            # variances_list[nl].append(np.var(np.sum(np.abs(full_gradient), axis=(1,2))) / param_count)
            # variances_list[nl].append(np.var(np.linalg.norm(full_gradient, 'fro', axis=(1,2))) / np.sqrt(param_count))
            # variances_list[nl].append(np.var(full_gradient[:, 0, -1]))
            variances_list[nl].append(np.var(full_gradient[:, 0, width-1]))
    return widths_range, norms_list, variances_list

def variances_plots():
    file_comp = data_folder + 'gradients/220711_qmio/gradients_nr0002.npz'
    file_gad3 = data_folder + 'gradients/220713_euler/gradients_merge_nr0001.npz'
    file_gad4 = data_folder + 'gradients/220716_euler/gradients_nr0001.npz'

    fig = abstract_formatter.figure(aspect_ratio=7/16, wide=True)
    ax = fig.subplots()

    qubits_list, norms_list, variances_list = get_vars_for_plot(file_comp)
    for line in range(len(variances_list)):
        normalized_variances = variances_list[line]/norms_list**2
        ax.semilogy(qubits_list, normalized_variances, ":s", c=palette[0][line])

    qubits_list, norms_list, variances_list = get_vars_for_plot(file_gad3)
    for line in range(len(variances_list)):
        normalized_variances = variances_list[line]/norms_list**2
        ax.semilogy(qubits_list, normalized_variances, "-o", c=palette[2][line])

    qubits_list, norms_list, variances_list = get_vars_for_plot(file_gad4)
    for line in range(len(variances_list)):
        normalized_variances = variances_list[line]/norms_list**2
        ax.semilogy(qubits_list, normalized_variances, "--o", c=palette[1][line])

    ax.set_xlabel(r"Number of computational qubits")
    ax.set_ylabel(r"$\mathrm{Var} \left[\partial \theta_{\nu} C \right]$")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    layers_list = np.load(file_comp)['layers_list']

    custom_lines = [Line2D([0], [0], ls=':', marker='s', c=palette[0][1], lw=1), 
                    Line2D([0], [0], ls='--', marker='o', c=palette[1][1], lw=1), 
                    Line2D([0], [0], ls='-', marker='o', c=palette[2][2], lw=1)]
    custom_lines += [Line2D([0], [0], ls='', c='k', lw=1)]
    custom_lines += [Line2D([0], [0], c=palette[-1][nl], lw=2) for nl in range(len(layers_list))]
    ax.legend(custom_lines, 
              ['comp', '4-local', '3-local'] + [''] + 
              ['{} layers'.format(num_layers) for num_layers in layers_list], 
              ncol=2)
    
    plt.tight_layout()
    plt.savefig(data_folder + '../plots/variances_new_gadget/variances_for_paper.pdf')

if __name__ == "__main__":
    # training_plots()
    variances_plots()