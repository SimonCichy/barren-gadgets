import datetime
import warnings
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


def scheduled_training(schedule, plot_data=True, save_data=False):
    """Training of a quantum circuit according to the provided schedule

    Args:
        schedule (dict)     : dictionnary containing all the necessary components for the training:
        - 'device'          : device to be used for the training
        - 'optimizers'      : list of optimizers to be used for the trainings
        - 'ansaetze'        : list of ansaetze to be used as trainable quantum 
                              circuit (e.g. StronglyEntanglingLayers or 
                              AlternatingLayeredAnsatz)
                              /!\ should have a method ansatz(params, wires)
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
    optimizer_list = schedule['optimizers']
    ansatz_list = schedule['ansaetze']
    weights = schedule['initial weights']
    training_obs = schedule['training observables']
    monitoring_obs = schedule['monitoring observables']
    label_list = ['Training cost'] + schedule['labels']
    max_iter_list = schedule['iterations']
    print_frequency = 100
    
    # ==========   Sanity checks   ==========
    # same number phases for all parameters
    assert len(training_obs) == len(max_iter_list)
    assert len(training_obs) == len(optimizer_list)
    assert len(training_obs) == len(ansatz_list)
    # all ansaetze have the correct width
    for nr, ans in enumerate(ansatz_list):
        if (np.shape(ans.gate_sequence)[1] != len(dev.wires)):
            warnings.warn('Ansatz nr. {} '.format(nr) + 
                          'does not act on the same number of qubits as the' + 
                          'device to run it on: ansatz on ' +
                          '{} qubits'.format(np.shape(ans.gate_sequence)[1]) + 
                          ' vs device with' + 
                          '{} wires'.format(len(dev.wires)))
        if (np.shape(weights)[1] != len(dev.wires)):
            warnings.warn('Initial weights expecting a different number of ' + 
                          'qubits than the device has: '
                          'weights width {}'.format(np.shape(weights)[1]) + 
                          ' vs device with' + 
                          '{} wires'.format(len(dev.wires)))
    
    # ==========   Setting up   ==========
    # Initializing the list of lists of cost functions to save
    cost_functions = [None] * (1 + len(monitoring_obs))
    cost_lists = [[] for _ in range(len(cost_functions))]
    # cost_lists = [[]] * len(cost_functions)  # /!\ all point to the same object
    # Defining the initial cost functions
    # ansatz = ansatz_list[0].ansatz
    # cost_functions = [qml.ExpvalCost(ansatz, training_obs[0], dev)]
    # cost_functions += [qml.ExpvalCost(ansatz, obs, dev) for obs in monitoring_obs]
    
    # ==========   Training   ==========
    # Looping through all the phases of the scheduled training (might be a 
    # change in trained observable, circuit depth, trainable gate set, ...)
    for phase in range(len(training_obs)):
        # updating the ansatz
        ansatz = ansatz_list[phase].ansatz
        # Checking if the ansatz has been deepened
        depth_difference = np.shape(ansatz_list[phase].gate_sequence)[0] - \
                           np.shape(weights)[0]
        if depth_difference < 0:
            warnings.warn('Ansatz depth has been shrunk. Please define a ' + 
                          'valid schedule. ' + 
                          'ansatz nr. {}: '.format(nr-1) + 
                          'of depth {} '.format(np.shape(weights)[0]) + 'vs ' +
                          'ansatz nr. {}: '.format(nr) + 
                          'of depth {} '.format(np.shape(ansatz.gate_sequence)[0]))
        elif depth_difference > 0:
            # padding with zeros to ensure "continuity" of the cost value
            # /!\ assumes that U(0) = II for all gates
            np.append(weights, np.zeros(depth_difference, len(dev.wires)))
        # updating the monitoring cost functions with the new ansatz
        cost_functions[0] = qml.ExpvalCost(ansatz, training_obs[phase], dev)
        for c, obs in enumerate(monitoring_obs):
            cost_functions[c+1] = qml.ExpvalCost(ansatz, obs, dev)
        # Adding the initial cost value for the new ansatz
        for c in range(len(cost_functions)):
            cost_lists[c].append(cost_functions[c](weights))
        training_cost = qml.ExpvalCost(ansatz, training_obs[phase], dev) # = cost_functions[0]
        # updating the training parameters
        max_iter = max_iter_list[phase]
        opt = optimizer_list[phase]
        print(f"Phase {phase:2d} | Training observable: ")
        print(training_obs[phase])
        print("Ansatz: ", type(ansatz_list[phase]).__name__)
        print("Iterations: ", max_iter)
        print(f"Iteration = {0:5d} of {max_iter:5d} | " +
              "Training cost = {:.8f} | ".format(cost_lists[0][-1]))
        # Looping through the iterations of the phase
        for it in range(max_iter):
            weights = opt.step(training_cost, weights)
            for c in range(len(cost_functions)):
                cost_lists[c].append(cost_functions[c](weights))
            if (it + 1) % print_frequency == 0:
                print(f"Iteration = {it+1:5d} of {max_iter:5d} | " +
                      "Training cost = {:.8f} | ".format(cost_lists[0][-1]))

    # ==========   Plotting   ==========
    if plot_data:
        training_iterations = list(range(max_iter_list[0]+1))
        max_iter_sum = max_iter_list[0]
        for max_iter in max_iter_list[1:]: 
            max_iter_sum += max_iter
            training_iterations += list(range(training_iterations[-1], max_iter_sum+1))
        plt.figure()
        for c, cost in enumerate(cost_lists):
            plt.plot(training_iterations, cost, label=label_list[c])
        plt.legend()

    # ==========    Saving    ==========
    if save_data:
        pass


