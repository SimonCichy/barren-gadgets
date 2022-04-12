import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import datetime

from gadget_cost import GadgetCost
from observables_holmes import ObservablesHolmes

np.random.seed(2)
data_folder = '../results/data/training/'
use_exact_ground_energy = False
plot_data = True
save_data = False
# cost_functions = ['global', 'local', 'gadget']
cost_functions = ['gadget']

computational_qubits = 6
ancillary_qubits = computational_qubits
num_layers = 2
max_iter = 50
gate_set = [qml.RX, qml.RY, qml.RZ]

perturbation_factor = 1
# locality = computational_qubits
# lambda_max = (locality - 1) / (4 * locality)
# l = perturbation_factor * lambda_max

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

# creating a cost function object
cfg = GadgetCost(computational_qubits, computational_qubits+ancillary_qubits, dev_gad)
cfc = GadgetCost(computational_qubits, computational_qubits, dev_comp)

# def cat_state_witness(params, gate_sequence, computational_qubits, device):
#     total_qubits = np.shape(params)[1]
#     gadget_qnode = qml.QNode(gadget_circuit, device)
#     cat_state = np.zeros(2**(total_qubits - computational_qubits))
#     cat_state[[0, -1]] = 1/np.sqrt(2) 
#     cat_projector = qml.Hermitian(np.outer(cat_state, cat_state), range(computational_qubits, total_qubits, 1))
#     return gadget_qnode(params, gate_sequence, computational_qubits, cat_projector)

# def ancillary_ground_projection(params, gate_sequence, computational_qubits, device):
#     total_qubits = np.shape(params)[1]
#     gadget_qnode = qml.QNode(gadget_circuit, device)
#     projector = np.zeros((2**(total_qubits - computational_qubits), 2**(total_qubits - computational_qubits)))
#     projector[[0, -1],[0, -1]] = 1
#     projector = qml.Hermitian(projector, range(computational_qubits, total_qubits, 1))
#     return gadget_qnode(params, gate_sequence, computational_qubits, projector)

# Global case:
if 'global' in cost_functions:
    print("Training the global cost function")
    weights_init = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits), requires_grad=True)
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits)] for _ in range(num_layers)]
    cost_computational = [cfc.cost_function(weights_init, random_gate_sequence, Hcomp)]

    weights = weights_init
    for it in range(max_iter):
        weights = opt.step(cfc.cost_function, weights, gate_sequence=random_gate_sequence, observable=Hcomp)
        cost_computational.append(cfc.cost_function(weights, random_gate_sequence, Hcomp))
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
        plt.plot(np.arange(max_iter+1), cost_computational)
    plt.show()

# Local case:
if 'local' in cost_functions:
    print("Training the local cost function")
    weights_init = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits), requires_grad=True)
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits)] for _ in range(num_layers)]
    cost_local = [cfc.cost_function(weights_init, random_gate_sequence, Hloc)]

    weights = weights_init
    for it in range(max_iter):
        weights = opt.step(cfc.cost_function, weights, gate_sequence=random_gate_sequence, observable=Hloc)
        cost_local.append(cfc.cost_function(weights, random_gate_sequence, Hcomp))
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
        plt.show()

# Gadget case:
if 'gadget' in cost_functions:
    print("Training the gadget cost function")
    # weights_init = np.zeros((num_layers, computational_qubits+ancillary_qubits), requires_grad=True)           # starting close to the ground state
    weights_init = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits+ancillary_qubits), requires_grad=True)
    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits+ancillary_qubits)] for _ in range(num_layers)]
    cost_computational = [cfg.cost_function(weights_init, random_gate_sequence, Hcomp)]
    cost_gadget = [cfg.cost_function(weights_init, random_gate_sequence, Hgad)]
    cost_ancillary = [cfg.cost_function(weights_init, random_gate_sequence, Hanc)]
    cost_perturbation = [cfg.cost_function(weights_init, random_gate_sequence, V)]
    # cat_witness = [cat_state_witness(weights_init, random_gate_sequence, computational_qubits, dev_gad)]
    # ground_witness = [ancillary_ground_projection(weights_init, random_gate_sequence, computational_qubits, dev_gad)]
    print(f"Iteration = {0:5d} | " +
        "Gadget cost = {:.8f} | ".format(cost_gadget[-1]) +
        "Computational cost = {:.8f}".format(cost_computational[-1]))

    weights = weights_init
    for it in range(max_iter):
        weights = opt.step(cfg.cost_function, weights, gate_sequence=random_gate_sequence, observable=Hgad)
        cost_computational.append(cfg.cost_function(weights, random_gate_sequence, Hcomp))
        cost_gadget.append(cfg.cost_function(weights, random_gate_sequence, Hgad))
        cost_ancillary.append(cfg.cost_function(weights, random_gate_sequence, Hanc))
        cost_perturbation.append(cfg.cost_function(weights, random_gate_sequence, V))
        # cat_witness.append(cat_state_witness(weights_init, random_gate_sequence, computational_qubits, dev_gad))
        # ground_witness.append(ancillary_ground_projection(weights_init, random_gate_sequence, computational_qubits, dev_gad))
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

