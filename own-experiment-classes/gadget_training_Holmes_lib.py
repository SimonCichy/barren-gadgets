import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import datetime

# from gadget_circuits import hardware_efficient_ansatz
from hardware_efficient_ansatz import HardwareEfficientAnsatz
from observables_holmes import ObservablesHolmes

np.random.seed(42)
data_folder = '../results/data/training/'
use_exact_ground_energy = False
plot_data = True
save_data = False
cost_functions = ['global', 'local', 'gadget']
# cost_functions = ['gadget']
# cost_functions = ['global']

computational_qubits = 6
ancillary_qubits = computational_qubits
num_layers = 2
max_iter = 500
gate_set = [qml.RX, qml.RY, qml.RZ]

perturbation_factor = 1

dev_comp = qml.device("default.qubit", wires=range(computational_qubits))
dev_gad = qml.device("default.qubit", wires=range(computational_qubits+ancillary_qubits))
opt = qml.GradientDescentOptimizer(stepsize=0.1)

# Used observables
oH = ObservablesHolmes(computational_qubits, ancillary_qubits, perturbation_factor)
Hcomp = oH.computational()
Hloc = oH.local()
Hanc = oH.ancillary()
V = oH.perturbation()
Hgad = oH.gadget()
Pcat = oH.cat_projector()
Pground = oH.ancillary_ground_projector()


# Global case:
if 'global' in cost_functions:
    print("Training the global cost function")
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits)] for _ in range(num_layers)]
    hea = HardwareEfficientAnsatz(random_gate_sequence)
    cost = qml.ExpvalCost(hea.ansatz, Hcomp, dev_comp)
    weights_init = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits), requires_grad=True)
    cost_computational = [cost(weights_init)]

    weights = weights_init
    for it in range(max_iter):
        weights = opt.step(cost, weights)
        cost_computational.append(cost(weights))
        # opt.update_stepsize(stepsize)
        if it % 50 == 0:
            print("Iteration = {:5d} | Cost function = {: .8f}".format(it+1, cost_computational[-1]))
        
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
        plt.title(r'$\langle \psi_{HE}| H^{comp} |\psi_{HE} \rangle$ for ' + r"$n_{comp}=$"+"{}".format(computational_qubits))
        plt.show()

# Local case:
if 'local' in cost_functions:
    print("Training the local cost function")
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits)] for _ in range(num_layers)]
    hea = HardwareEfficientAnsatz(random_gate_sequence)
    cost = qml.ExpvalCost(hea.ansatz, Hloc, dev_comp)
    weights_init = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits), requires_grad=True)
    cost_local = [cost(weights_init)]

    weights = weights_init
    for it in range(max_iter):
        weights = opt.step(cost, weights)
        cost_local.append(cost(weights))
        if it % 50 == 0:
            print("Iteration = {:5d} | Cost function = {: .8f}".format(it+1, cost_local[-1]))

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
        plt.title(r'$\langle \psi_{HE}| H^{loc} |\psi_{HE} \rangle$ for ' + r"$n_{comp}=$"+"{}".format(computational_qubits))
        plt.show()

# Gadget case:
if 'gadget' in cost_functions:
    print("Training the gadget cost function")
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits+ancillary_qubits)] for _ in range(num_layers)]
    hea = HardwareEfficientAnsatz(random_gate_sequence)
    cost_comp = qml.ExpvalCost(hea.ansatz, Hcomp, dev_gad)
    cost_gad = qml.ExpvalCost(hea.ansatz, Hgad, dev_gad)
    cost_anc = qml.ExpvalCost(hea.ansatz, Hanc, dev_gad)
    cost_pert = qml.ExpvalCost(hea.ansatz, V, dev_gad)
    # cost_wit = qml.ExpvalCost(hea.ansatz, Pcat, dev_gad)
    # weights_init = np.zeros((num_layers, computational_qubits+ancillary_qubits), requires_grad=True)           # starting close to the ground state
    weights_init = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits+ancillary_qubits), requires_grad=True)
    cost_computational = [cost_comp(weights_init)]
    cost_gadget = [cost_gad(weights_init)]
    cost_ancillary = [cost_anc(weights_init)]
    cost_perturbation = [cost_pert(weights_init)]
    # cat_witness = [cost_wit(weights_init)]
    # ground_witness = [cfg.cost_function(weights_init, random_gate_sequence, Pground)]
    print(f"Iteration = {0:5d} | " +
        "Gadget cost = {:.8f} | ".format(cost_gadget[-1]) +
        "Computational cost = {:.8f}".format(cost_computational[-1]))

    weights = weights_init
    for it in range(max_iter):
        weights = opt.step(cost_gad, weights)
        cost_computational.append(cost_comp(weights))
        cost_gadget.append(cost_gad(weights))
        cost_ancillary.append(cost_anc(weights))
        cost_perturbation.append(cost_pert(weights))
        # cat_witness.append(cost_wit(weights_init))
        # ground_witness.append(cfg.cost_function(weights_init, random_gate_sequence, Pground))
        # opt.update_stepsize(stepsize)
        if it % 10 == 0:
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
        # p_plus, = ax2.plot(np.arange(max_iter+1), cat_witness, '--', c='salmon', label=r'$|\langle \psi_{HE}| +\rangle |^2 $')
        # p_plus, = ax2.plot(np.arange(max_iter+1), ground_witness, '--', c='salmon', label=r'$|\langle \psi_{HE}| +\rangle |^2 $')
        # p = [p_gad, p_comp, p_anc, p_pert, p_plus]
        p = [p_gad, p_comp, p_anc, p_pert]
        ax.set_xlabel(r"Number of iterations")
        ax.set_ylabel(r"Cost function")
        ax.tick_params(axis ='y', labelcolor = 'navy')
        ax2.tick_params(axis ='y', labelcolor = 'maroon')
        ax.legend(p, [p_.get_label() for p_ in p])
        ax.set_title(r"$n_{comp}=$"+"{}".format(computational_qubits))
        plt.show()
    
    if save_data:
        with open(data_folder + '{}_training_gadget_{}qubits_{}layers_{}iterations_{}lambda.dat'
                    .format(datetime.datetime.now().strftime("%y%m%d"),
                            computational_qubits, num_layers, max_iter, perturbation_factor), 'w') as of:
            of.write('# iteration\tcost gadget\tcost computational\tcost ancillary\tcost perturbation\tcat projection\n')

            for it in range(max_iter+1):
                # of.write('{}\t{}\t{}\t{}\t{}\n'.format(it, cost_gadget[it], cost_computational[it], cost_ancillary[it], cost_perturbation[it], cat_witness[it]))
                of.write('{}\t{}\t{}\t{}\t{}\n'.format(it, cost_gadget[it], cost_computational[it], cost_ancillary[it], cost_perturbation[it]))

