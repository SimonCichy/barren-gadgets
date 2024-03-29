import sys
sys.path.append('../src')
sys.path.append('src')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np
from data_management import get_training_costs, get_training_labels2

data_folder = '../results/data/'

palette = [['#e65e71', '#d64055', '#ba182e', '#851919'], 
           ['#77f2d8', '#64d4bc', '#3cb89d', '#279680'], 
           ['#00a1d9', '#1977c2', '#2041ba', '#202482'], 
           ['#b0b0b0', '#727272', '#606060', '#393939']]

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams.update({
    "pgf.preamble": "\n".join([
        r"\usepackage{amsmath}", 
        r"\usepackage{amssymb}", 
        # r"\usepackage{dsfont}", 
        r"\usepackage{xcolor}"
    ])
})
plt.rcParams["figure.figsize"] = 23 * 0.3937, 23 * 0.7 * 0.3937

def training_plots(training='plain'): 
    file_list = []
    if training == 'plain':
        file_list  += ['220726_qmio11-1/training_nr{:0>4}'.format(nr) for nr in range(3, 11, 3)]
        file_list  += ['220726_qmio14/training_nr{:0>4}'.format(nr) for nr in range(3, 11, 3)]
        file_list  += ['220727_qmio11-1/training_nr{:0>4}'.format(nr) for nr in range(2, 21, 3)]
        file_list  += ['220727_qmio11-2/training_nr{:0>4}'.format(nr) for nr in range(3, 31, 3)]
        file_list  += ['220727_qmio14/training_nr{:0>4}'.format(nr) for nr in range(2, 33, 3)]
        file_list  += ['220728_qmio11-1/training_nr{:0>4}'.format(nr) for nr in range(3, 20, 3)]
        file_list  += ['220728_qmio11-2/training_nr{:0>4}'.format(nr) for nr in range(3, 20, 3)]
        file_list  += ['220728_qmio14/training_nr{:0>4}'.format(nr) for nr in range(3, 40, 3)]
        file_list  += ['220729_qmio11-1/training_nr{:0>4}'.format(nr) for nr in range(2, 12, 3)]
        file_list  += ['220729_qmio11-2/training_nr{:0>4}'.format(nr) for nr in range(2, 12, 3)]
        file_list  += ['220729_qmio14/training_nr{:0>4}'.format(nr) for nr in range(3, 10, 3)]
    elif training == 'reordered':
        file_list  = ['220831/training_nr{:0>4}'.format(nr) for nr in range(3, 17, 3)]
        file_list += ['220901/training_nr{:0>4}'.format(nr) for nr in range(2, 47, 3)]
        file_list += ['220902/training_nr{:0>4}'.format(nr) for nr in range(1, 29, 3)]
    
    palette_choice = [palette[0][2], palette[2][0]]

    fig = plt.figure()
    ax = fig.subplots()

    iterations = 500
    runs = len(file_list)
    print("perturbation factor: {:>6.1f}, number of trainings: {:>4}".format(10, runs))
    ax.plot(np.ones(iterations), ':', c='gainsboro')
    ax.plot(-np.ones(iterations), ':', c='gainsboro')
    ax.plot(np.zeros(iterations), ':', c='gainsboro')
    # plt.set_title(r'$\lambda = {}$'.format(lambdas[l]) + r'$\lambda_{max}$', 
    #                  fontsize=9)
    ax.set_ylim(top=2)
    ax.set_ylabel('Cost', fontsize = 20)
    ax.set_xlabel('Number of iterations', fontsize = 20)
    costs_sum = 0
    for f, file in enumerate(file_list):
        file = data_folder + 'training/' + file
        costs = get_training_costs(file + '.npz')
        # labels = get_training_labels2(file + '.txt')
        costs_sum += costs
        for i in range(1, 3):
            ax.plot(costs[i],'-', c=palette_choice[i-1], alpha=1.5/runs)
    costs_mean = costs_sum / runs
    for i in range(1, 3):
        ax.plot(costs_mean[i],'-', c=palette_choice[i-1])
    custom_lines = [Line2D([0], [0], color=palette_choice[1-nl], lw=1) for nl in range(2)]
    ax.legend(handles=custom_lines,
                labels=[r'$ \big \langle H^\mathrm{gad} \big\rangle_{| \psi(\mathbf{\theta})\rangle}$', 
                        r'$ \big \langle H^\mathrm{comp}\otimes{1}\!\mathrm{I}^\mathrm{anc} \big\rangle_{| \psi(\mathbf{\theta})\rangle} $'], 
                fontsize = 20, 
                handlelength=.8, 
                loc='upper right',
                bbox_to_anchor=(1, 0.96),
                edgecolor='1', 
                borderpad=0)
    plt.tight_layout()
    plt.show()

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
    file_gad3 = data_folder + 'gradients/220725_euler/gradients_nr0002_merge.npz'
    file_gad4 = data_folder + 'gradients/220728_euler/gradients_nr0001_merge.npz'

    fig = plt.figure()
    ax = fig.subplots()
    ax.set_xlabel(r"{ Number of computational qubits}", labelpad=0, fontsize = 16)
    ax.set_ylabel(r"${ \mathrm{Var} \left[\partial C / \partial \mathbf{\theta}_{\nu} \right]}$", fontsize = 16)

    qubits_list, norms_list, variances_list = get_vars_for_plot(file_comp)
    for line in range(len(variances_list)):
        normalized_variances = variances_list[line]/norms_list**2
        ax.loglog(qubits_list, normalized_variances, ":s", c=palette[0][line])

    qubits_list, norms_list, variances_list = get_vars_for_plot(file_gad3)
    for line in range(len(variances_list)):
        normalized_variances = variances_list[line]/norms_list**2
        ax.loglog(qubits_list, normalized_variances, "-o", c=palette[2][line])

    qubits_list, norms_list, variances_list = get_vars_for_plot(file_gad4)
    for line in range(len(variances_list)):
        normalized_variances = variances_list[line]/norms_list**2
        ax.loglog(qubits_list, normalized_variances, "--o", c=palette[1][line])

    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.ticklabel_format(axis="x", style='plain')

    layers_list = np.load(file_comp)['layers_list']
    custom_lines = [Line2D([0], [0], ls=':', marker='s', c=palette[0][1], lw=1), 
                    Line2D([0], [0], ls='--', marker='o', c=palette[1][1], lw=1), 
                    Line2D([0], [0], ls='-', marker='o', c=palette[2][2], lw=1)]
    custom_lines += [Line2D([0], [0], ls='', c='k', lw=1)]
    custom_lines += [Line2D([0], [0], c=palette[-1][nl], lw=2) for nl in range(len(layers_list))]
    ax.legend(handles=custom_lines, 
              labels=[r'$H^\mathrm{comp}$', '4-local ' + r'$H^\mathrm{gad}$', '3-local ' + r'$H^\mathrm{gad}$'] + [''] + 
                     [r'$\ \,$' + '{} layers'.format(num_layers) for num_layers in layers_list[:2]] + 
                     ['{} layers'.format(num_layers) for num_layers in layers_list[2:]], 
              loc='upper right',
              ncol=2, 
              fontsize = 16,
              handletextpad=1, 
              handlelength=2, 
              labelspacing=.18, 
              frameon=False)
    ax.spines.right.set_visible(True)
    ax.spines.top.set_visible(True)
    plt.tight_layout()
    # plt.savefig(data_folder + '../plots/variances_new_gadget/variances_for_presentation.png')
    plt.show()

if __name__ == "__main__":
    training_plots('plain')
    # variances_plots()