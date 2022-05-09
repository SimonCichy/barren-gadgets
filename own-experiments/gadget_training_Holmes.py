import sys
sys.path.append('../src')
sys.path.append('src')
import warnings
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import datetime

# from hardware_efficient_ansatz import HardwareEfficientAnsatz
from hardware_efficient_ansatz import AlternatingLayeredAnsatz
from observables_holmes import ObservablesHolmes

seed = 2
np.random.seed(seed)
data_folder = '../results/data/training/'
use_exact_ground_energy = False
plot_data = True
save_data = False
# cost_functions = ['global', 'local', 'gadget']
cost_functions = ['gadget']

computational_qubits = 4
ancillary_qubits = int(1 * computational_qubits)
num_layers = 8
max_iter = 2000
step = 0.3
print_frequency = 100
num_shots = 1000

gate_set = [qml.RX, qml.RY, qml.RZ]
perturbation_factors = np.linspace(0, 1, 6)

dev_comp = qml.device("default.qubit", 
                      wires=range(computational_qubits), 
                      shots=num_shots)
dev_gad = qml.device("default.qubit", 
                     wires=range(computational_qubits+ancillary_qubits), 
                     shots=num_shots)
opt = qml.GradientDescentOptimizer(stepsize=step)


# Global case:
def training_global(observable_generator, 
                    max_iterations = 100, 
                    plot_data=True, 
                    save_data=False):
    Hcomp = observable_generator.computational()
    random_gate_sequence = [[np.random.choice(gate_set) 
                             for _ in range(computational_qubits)] 
                             for _ in range(num_layers)]
    ala = AlternatingLayeredAnsatz(random_gate_sequence)
    cost = qml.ExpvalCost(ala.ansatz, Hcomp, dev_comp)
    weights_init = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits), requires_grad=True)
    cost_computational = [cost(weights_init)]

    weights = weights_init
    for it in range(max_iterations):
        weights = opt.step(cost, weights)
        cost_computational.append(cost(weights))
        # opt.update_stepsize(stepsize)
        if it % 50 == 0:
            print("Iteration = {:5d} | ".format(it+1) + 
                  "Cost function = {: .8f}".format(cost_computational[-1]))
        
    if save_data: 
        with open(data_folder + '{}_training_global_{}qubits_{}layers_{}iterations.dat'
                    .format(datetime.datetime.now().strftime("%y%m%d"),
                            computational_qubits, num_layers, max_iter), 'w') as of:
            of.write('# iteration\tcost\n')

            for it in range(max_iter+1):
                of.write('{}\t{}\n'.format(it, cost_computational[it]))

    if plot_data:
        plt.figure()
        plt.plot(np.arange(max_iter+1), cost_computational, c='firebrick')
        plt.xlabel(r"Number of iterations")
        plt.ylabel(r"Cost function")
        plt.title(r'$\langle \psi_{HE}| H^{comp} |\psi_{HE} \rangle$ for ' + 
                  r"$n_{comp}=$"+"{}".format(computational_qubits))
        plt.show()

# Local case:
def training_local(observable_generator, 
                   max_iterations = 100, 
                   plot_data=True, 
                   save_data=False):
    Hloc = observable_generator.local()
    random_gate_sequence = [[np.random.choice(gate_set) 
                             for _ in range(computational_qubits)] 
                             for _ in range(num_layers)]
    ala = AlternatingLayeredAnsatz(random_gate_sequence)
    cost = qml.ExpvalCost(ala.ansatz, Hloc, dev_comp)
    weights_init = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits), requires_grad=True)
    cost_local = [cost(weights_init)]

    weights = weights_init
    for it in range(max_iterations):
        weights = opt.step(cost, weights)
        cost_local.append(cost(weights))
        if it % 50 == 0:
            print("Iteration = {:5d} | ".format(it+1) + 
                  "Cost function = {: .8f}".format(cost_local[-1]))

    if save_data:
        with open(data_folder + '{}_training_local_{}qubits_{}layers_{}iterations.dat'
                    .format(datetime.datetime.now().strftime("%y%m%d"),
                            computational_qubits, num_layers, max_iter), 'w') as of:
            of.write('# iteration\tcost\n')

            for it in range(max_iter+1):
                of.write('{}\t{}\n'.format(it, cost_local[it]))

    if plot_data:
        plt.figure()
        plt.plot(np.arange(max_iter+1), cost_local)
        plt.xlabel(r"Number of iterations")
        plt.ylabel(r"Cost function")
        plt.title(r'$\langle \psi_{HE}| H^{loc} |\psi_{HE} \rangle$ for ' + 
                  r"$n_{comp}=$"+"{}".format(computational_qubits))
        plt.show()

# Gadget case:
def training_gadget(observable_generator, l_factor= 0.5, max_iterations = 100, 
                    gate_sequence=None, initial_weights=None, 
                    plot_data=True, save_data=False, print_frequency=10, 
                    check_witness=False):
    
    # Used observables
    Hcomp = observable_generator.computational()
    Hanc = observable_generator.ancillary()
    V = observable_generator.perturbation()
    Hgad = observable_generator.gadget()
    Pcat = observable_generator.cat_projector()
    # Pground = observable_generator.ancillary_ground_projector()
    Pground_comp = observable_generator.computational_ground_projector()

    if gate_sequence is None:
        # Random initialization
        gate_sequence = [[np.random.choice(gate_set) 
                          for _ in range(computational_qubits+ancillary_qubits)] 
                          for _ in range(num_layers)]
    ala = AlternatingLayeredAnsatz(gate_sequence)
    if initial_weights is None:
        # weights_init = np.zeros((num_layers, computational_qubits+ancillary_qubits), requires_grad=True)           # starting close to the ground state
        initial_weights = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits+ancillary_qubits), requires_grad=True)

    # Cost function definitions
    cost_comp = qml.ExpvalCost(ala.ansatz, Hcomp, dev_gad)
    cost_gad = qml.ExpvalCost(ala.ansatz, Hgad, dev_gad)
    cost_anc = qml.ExpvalCost(ala.ansatz, Hanc, dev_gad)
    cost_pert = qml.ExpvalCost(ala.ansatz, V, dev_gad)
    if check_witness:
        proj_cat = qml.ExpvalCost(ala.ansatz, Pcat, dev_gad)
        proj_gs = qml.ExpvalCost(ala.ansatz, Pground_comp, dev_gad)

    # Initial cost values
    cost_computational = [cost_comp(initial_weights)]
    cost_gadget = [cost_gad(initial_weights)]
    cost_ancillary = [cost_anc(initial_weights)]
    cost_perturbation = [cost_pert(initial_weights)]
    if check_witness:
        cat_witness = [proj_cat(initial_weights)]
        gs_witness = [proj_gs(initial_weights)]
    print(f"Iteration = {0:5d} | " +
        "Gadget cost = {:.8f} | ".format(cost_gadget[-1]) +
        "Computational cost = {:.8f}".format(cost_computational[-1]))

    weights = initial_weights
    for it in range(max_iterations):
        weights = opt.step(cost_gad, weights)
        cost_computational.append(cost_comp(weights))
        cost_gadget.append(cost_gad(weights))
        cost_ancillary.append(cost_anc(weights))
        cost_perturbation.append(cost_pert(weights))
        if check_witness:
            cat_witness.append(proj_cat(weights))
            gs_witness.append(proj_gs(weights))
        # opt.update_stepsize(stepsize)
        if (it + 1) % print_frequency == 0:
            print(f"Iteration = {it+1:5d} | " +
                "Gadget cost = {:.8f} | ".format(cost_gadget[-1]) +
                "Computational cost = {:.8f}".format(cost_computational[-1]))

    if plot_data:
        fig, ax = plt.subplots()
        ax2 = ax.twinx() 
        p_gad, = ax.plot(np.arange(max_iter+1), cost_gadget, c='mediumblue', label=r'$\langle \psi_{HE}| H^{gad} |\psi_{HE} \rangle$')
        p_comp, = ax2.plot(np.arange(max_iter+1), cost_computational, c='firebrick', label=r'$\langle \psi_{HE}| H^{comp} |\psi_{HE} \rangle$')
        p_anc, = ax.plot(np.arange(max_iter+1), cost_ancillary, ':', c='royalblue', label=r'$\langle \psi_{HE}| H^{anc} |\psi_{HE} \rangle$')
        p_pert, = ax.plot(np.arange(max_iter+1), cost_perturbation, ':', c='darkturquoise', label=r'$\langle \psi_{HE}| \lambda V |\psi_{HE} \rangle$') 
        p = [p_gad, p_comp, p_anc, p_pert]
        if check_witness:
            p_plus, = ax2.plot(np.arange(max_iter+1), cat_witness, '--', c='salmon', label=r'$|\langle \psi_{HE}| +\rangle |^2 $')
            # p_plus, = ax2.plot(np.arange(max_iter+1), ground_witness, '--', c='salmon', label=r'$|\langle \psi_{HE}| +\rangle |^2 $')
            p_gs, = ax2.plot(np.arange(max_iter+1), gs_witness, '--', c='salmon', label=r'$|\langle \psi_{HE}| P_{gs}^{comp}| \psi_{HE} \rangle |^2 $')
            p += [p_plus, p_gs]

        ax.set_xlabel(r"Number of iterations")
        ax.set_ylabel(r"Cost function")
        ax.tick_params(axis ='y', labelcolor = 'navy')
        ax2.tick_params(axis ='y', labelcolor = 'maroon')
        ax.legend(p, [p_.get_label() for p_ in p])
        ax.set_title(r"$n_{comp}=$" + "{}".format(computational_qubits) + 
                     r", $\lambda=$" + "{:1.1f}".format(l_factor))
        # plt.show()
    
    if save_data:
        locality = observable_generator.loc
        subfolder = 'gadget{}/'.format(locality)
        with open(data_folder + subfolder + 
                  '{}_'.format(datetime.datetime.now().strftime("%y%m%d")) + 
                  'training_gadget{}_'.format(locality) + 
                  '{:02}qubits_'.format(computational_qubits) + 
                  '{:02}layers_'.format(num_layers) + 
                  '{}iterations_'.format(max_iter) + 
                  'step{}_'.format(step) + 
                  'seed{:02}_'.format(seed) + 
                  '{:1.1f}lambda.dat'.format(l_factor), 'w') as of:
            of.write('# iteration\t' + 'cost computational\t' + 
                     'cost gadget\t' + 'cost ancillary\t' + 
                     'cost perturbation\t' + 'cat projection\t' + 
                     'computational ground projection\n')

            for it in range(max_iter+1):
                if check_witness:
                    of.write('{}\t'.format(it) + 
                             '{}\t'.format(cost_computational[it]) + 
                             '{}\t'.format(cost_gadget[it]) + 
                             '{}\t'.format(cost_ancillary[it]) + 
                             '{}\t'.format(cost_perturbation[it]) + 
                             '{}\t'.format(cat_witness[it]) +
                             '{}\n'.format(gs_witness[it]))
                else:
                    of.write('{}\t'.format(it) + 
                             '{}\t'.format(cost_computational[it]) + 
                             '{}\t'.format(cost_gadget[it]) + 
                             '{}\t'.format(cost_ancillary[it]) + 
                             '{}\n'.format(cost_perturbation[it]))


if __name__ == "__main__":
    if 'global' in cost_functions:
        print("Training the global cost function")
        oH = ObservablesHolmes(computational_qubits, 0, 0)
        training_global(observable_generator=oH, max_iterations=max_iter, plot_data=plot_data, save_data=save_data)
    if 'local' in cost_functions:
        print("Training the local cost function")
        oH = ObservablesHolmes(computational_qubits, 0, 0)
        training_local(observable_generator=oH, max_iterations=max_iter, plot_data=plot_data, save_data=save_data)
    if 'gadget' in cost_functions:
        print("Training the gadget cost function")
        print(" Computational qubits:   ", computational_qubits)
        print(" Ancillary qubits:       ", ancillary_qubits)
        print(" Layers:                 ", num_layers)
        print(" Iterations:             ", max_iter)
        print(" Random seed:            ", seed)
        random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits+ancillary_qubits)] for _ in range(num_layers)]
        initial_weights = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits+ancillary_qubits), requires_grad=True)
        for pf in perturbation_factors:
            print(" Perturbation factor:    ", pf)
            oH = ObservablesHolmes(computational_qubits, ancillary_qubits, pf)
            training_gadget(observable_generator=oH, l_factor=pf, 
                            max_iterations=max_iter, 
                            gate_sequence=random_gate_sequence, 
                            initial_weights=initial_weights, 
                            plot_data=plot_data, save_data=save_data, 
                            print_frequency=print_frequency, 
                            check_witness=True)
    plt.show()
