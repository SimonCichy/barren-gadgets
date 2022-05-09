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

seed = 4
np.random.seed(seed)
data_folder = '../results/data/training/'
use_exact_ground_energy = False
plot_data = True
save_data = True
# cost_functions = ['global', 'local', 'gadget']
cost_functions = ['gadget']

computational_qubits = 4
ancillary_qubits = int(1 * computational_qubits)
num_layers = 8
max_iter = 5000
step = 0.3
print_frequency = 100

gate_set = [qml.RX, qml.RY, qml.RZ]
perturbation_factors = np.linspace(0, 1, 6)

dev_comp = qml.device("default.qubit", wires=range(computational_qubits))
dev_gad = qml.device("default.qubit", wires=range(computational_qubits+ancillary_qubits))
opt = qml.GradientDescentOptimizer(stepsize=step)



def scheduled_training(schedule):
    """Training of a quantum circuit according to the provided schedule

    Args:
        schedule (dict)     : dictionnary containing all the necessary components for the training:
        - 'device'          : device to be used for the training
        - 'optimizers'      : list of optimizers to be used for the trainings
        - 'ansaetze'        : list of ansaetze to be used as trainable quantum 
                              circuit (e.g. StronglyEntanglingLayers or 
                              AlternatingLayeredAnsatz)
                              /!\ it is responsibility of the user to ensure the
                              'continuity' of consecutive ansaetze s.t. the 
                              trained parameters can be used on consecutive ones
        - 'initial weights' : weights with which to start the training
        - 'training observables'    : list of observables to train the circuit on
        - 'monitoring observables'  : list of observables to keep track of during training
        - 'labels'                  : list of labels describing the monitored observables
        - 'iterations'              : list of number of iterations to be training on each of the observables

    Returns:
        plots
        saved files
    """
    # Getting the settings from the schedule dictionary
    dev = schedule['device']
    ansatz_list = schedule['ansaetze']
    optimizer_list = schedule['optimizers']
    weights = schedule['initial weights']
    training_obs = schedule['training observables']
    monitoring_obs = schedule['monitoring observables']
    label_list = schedule['labels']
    max_iter_list = schedule['iterations']
    # Sanity checks
    for nr, ansatz in enumerate(ansatz_list):
        if np.shape(ansatz.gate_sequence) != np.shape(weights):
            warnings.warn('weights does not have the dimension expected by ' + \
                          'ansatz nr. {}: '.format(nr) + \
                          'weights of shape {}'.format(np.shape(weights)) + ' vs '\
                          'gate sequece of shape {}'.format(np.shape(ansatz.gate_sequence)))
    assert len(training_obs) == len(max_iter_list)
    
    # Defining the cost functions
    cost_functions = [qml.ExpvalCost(ansatz, training_obs[0], dev)]
    cost_functions += [qml.ExpvalCost(ansatz, obs, dev) for obs in monitoring_obs]
    # Initializing the list of lists of cost functions to save
    cost_lists = [] * len(monitoring_obs)
    # Adding the initial cost value for each for the initial weights
    for c in range(len(cost_functions)):
        cost_lists[c].append(cost_functions[c](weights))
    
    # ==========   Training   ==========
    # Looping through all the phases of the scheduled training (might be a 
    # change in trained observable, circuit depth, trainable gate set, ...)
    for phase in len(training_obs):
        # updating the monitoring cost functions with the new ansatz
        cost_functions[0] = qml.ExpvalCost(ansatz, training_obs[phase], dev)
        for c, obs in enumerate(monitoring_obs):
            cost_functions[c+1] = qml.ExpvalCost(ansatz, obs, dev)
        print(f"Phase {phase:2d} | Training observable: ")
        print(training_obs[phase])
        # updating the training parameters
        max_iter = max_iter_list[phase]
        training_cost = qml.ExpvalCost(ansatz, training_obs[phase], dev)
        opt = optimizer_list[phase]
        print(f"Iteration = {0:5d} of {max_iter:5d} | " +
              "Training cost = {:.8f} | ".format(cost_lists[0][-1]))
        # Looping through the iterations of the phase
        for it in range(max_iter):
            weights = opt.step(training_cost, weights)
            for c in range(len(cost_functions)):
                cost_lists[c].append(cost_functions[c](weights))
            if it % print_frequency == 0:
                print(f"Iteration = {it+1:5d} of {max_iter:5d} | " +
                      "Training cost = {:.8f} | ".format(cost_lists[0][-1]))
            pass
    # ==========   Plotting   ==========
    if plot_data:
        pass
    # ==========    Saving    ==========
    if save_data:
        pass

if __name__ == "__main__":
    # if 'global' in cost_functions:
    #     print("Training the global cost function")
    #     oH = ObservablesHolmes(computational_qubits, 0, 0)
    #     training_global(observable_generator=oH, max_iterations=max_iter, plot_data=plot_data, save_data=save_data)
    # if 'local' in cost_functions:
    #     print("Training the local cost function")
    #     oH = ObservablesHolmes(computational_qubits, 0, 0)
    #     training_local(observable_generator=oH, max_iterations=max_iter, plot_data=plot_data, save_data=save_data)
    # if 'gadget' in cost_functions:
    #     print("Training the gadget cost function")
    #     print(" Computational qubits:   ", computational_qubits)
    #     print(" Ancillary qubits:       ", ancillary_qubits)
    #     print(" Layers:                 ", num_layers)
    #     print(" Iterations:             ", max_iter)
    #     print(" Random seed:            ", seed)
    #     random_gate_sequence = [[np.random.choice(gate_set) for _ in range(computational_qubits+ancillary_qubits)] for _ in range(num_layers)]
    #     initial_weights = np.random.uniform(0, np.pi, size=(num_layers, computational_qubits+ancillary_qubits), requires_grad=True)
    #     for pf in perturbation_factors:
    #         print(" Perturbation factor:    ", pf)
    #         oH = ObservablesHolmes(computational_qubits, ancillary_qubits, pf)
    #         training_gadget(observable_generator=oH, l_factor=pf, 
    #                         max_iterations=max_iter, 
    #                         gate_sequence=random_gate_sequence, 
    #                         initial_weights=initial_weights, 
    #                         plot_data=plot_data, save_data=save_data, 
    #                         print_frequency=print_frequency, 
    #                         check_witness=True)
    plt.show()
