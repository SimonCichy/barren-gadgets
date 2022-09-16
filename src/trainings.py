import warnings
import time
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from hardware_efficient_ansatz import AlternatingLayeredAnsatz, SimplifiedAlternatingLayeredAnsatz
from observables_holmes import ObservablesHolmes
from jordan_gadgets import PerturbativeGadgets
from faehrmann_gadgets import NewPerturbativeGadgets
from data_management import save_training2

#TODO: remove any dependency on ObservableHolmes --> replace oH.comp() with 
# qml.Hamiltonian([1], [qml.operation.Tensor(*[qml.PauliZ(q) for q in range(n_comp)])])
#TODO: eliminate the schedules based on PerturbativeGadgets (deprecated),
# eliminate jordan_gadgets all together and rename jordan_gadgets

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
    # start timer
    tic = time.perf_counter()
    # Getting the settings from the schedule dictionary
    dev = schedule['device']
    optimizer_list = schedule['optimizers']
    ansatz_list = schedule['ansaetze']
    weights = schedule['initial weights']
    training_obs = schedule['training observables']
    monitoring_obs = schedule['monitoring observables']
    label_list = ['Training cost'] + schedule['labels']
    max_iter_list = [int(i) for i in schedule['iterations']]
    print_frequency = 20
    
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
    if save_data:
        toc = time.perf_counter()
        save_training2(schedule, cost_lists, mode='new file', runtime=toc-tic)
    
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
                          'ansatz nr. {}: of depth '.format(nr) + 
                          '{}'.format(np.shape(ansatz_list[phase].gate_sequence)[0]))
        elif depth_difference > 0:
            # padding with zeros to ensure "continuity" of the cost value
            # /!\ assumes that U(0) = II for all gates
            #TODO: check that these angles are then trained and don't stay at 0
            np.append(weights, np.zeros((depth_difference, len(dev.wires))))
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
              "Training cost = {:12.8f} | ".format(cost_lists[0][-1]) +
              "Monitoring cost = {:12.8f} | ".format(cost_lists[1][-1]))
        # Looping through the iterations of the phase
        for it in range(max_iter):
            weights = opt.step(training_cost, weights)
            for c in range(len(cost_functions)):
                cost_lists[c].append(cost_functions[c](weights))
            if (it + 1) % print_frequency == 0:
                print(f"Iteration = {it+1:5d} of {max_iter:5d} | " +
                      "Training cost = {:12.8f} | ".format(cost_lists[0][-1]) +
                      "Monitoring cost = {:12.8f} | ".format(cost_lists[1][-1]))
                if save_data:
                    toc = time.perf_counter()
                    save_training2(schedule, cost_lists, mode='overwrite', runtime=toc-tic)

    # ==========    Saving    ==========
    if save_data:
        toc = time.perf_counter()
        save_training2(schedule, cost_lists, mode='overwrite', runtime=toc-tic)
    
    # ==========   Plotting   ==========
    if plot_data:
        training_iterations = list(range(max_iter_list[0]+1))
        max_iter_sum = max_iter_list[0]
        for max_iter in max_iter_list[1:]: 
            max_iter_sum += max_iter
            training_iterations += list(range(training_iterations[-1], max_iter_sum+1, 1))
        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()
        ax2.plot(training_iterations, cost_lists[0], '--', c='grey', label=label_list[0])
        ax1.plot(training_iterations, np.ones(len(training_iterations)), ':', c='gainsboro')
        ax1.plot(training_iterations, -np.ones(len(training_iterations)), ':', c='gainsboro')
        ax1.plot(training_iterations, np.zeros(len(training_iterations)), ':', c='gainsboro')
        for c, cost in enumerate(cost_lists[1:]):
            ax1.plot(training_iterations, cost, label=label_list[c+1])
        ax1.legend()
        ax2.set_ylabel('Training cost', color='grey')
        ax2.tick_params(axis ='y', labelcolor='grey')


class SchedulesOfInterest:
    """ Class serving as a repository of used schedules that are interesting"""

    def __init__(self, computational_qubits, 
                 seed, num_shots=None):
        self.n_comp = int(computational_qubits)
        self.gate_set = [qml.RX, qml.RY, qml.RZ]
        self.np_rdm_seed = seed
        self.num_shots = num_shots

    def linear_ala_gad(self, perturbation, optimizer, iterations, target_locality=2):
        warnings.warn("Training schedule using Jordan and Fahri's gadget " + 
                      "construction. That construction has no guarantees of " +
                      "having the right low-energy spectrum for VQEs. " + 
                      "Use the new gadget construction instead.")
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        Hcomp = oH.computational()
        gadgetizer = PerturbativeGadgets(method='Jordan', 
                                         perturbation_factor=perturbation)
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality)
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = r * int(k / (target_locality - 1))
        num_layers = self.n_comp + n_anc
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp + n_anc)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        schedule = {
            'name': 'linear_ala_gad',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer], 
            'seed': self.np_rdm_seed,
            'ansaetze': [ala],
            'initial weights': initial_weights, 
            'training observables': [Hgad],
            'monitoring observables': [Hcomp, 
                                    #    oH.ancillary(), 
                                    #    oH.perturbation(), 
                                       Hgad,  
                                       oH.computational_ground_projector(), 
                                       gadgetizer.cat_projector(Hcomp, target_locality),
                                       gadgetizer.ancillary_X(Hcomp, target_locality)],
            'labels': [r'$\langle \psi_{ALA}| H^{comp} |\psi_{ALA} \rangle$', 
                    #    r'$\langle \psi_{ALA}| H^{anc} |\psi_{ALA} \rangle$', 
                    #    r'$\langle \psi_{ALA}| \lambda V |\psi_{ALA} \rangle$',
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| H^{gad}]$', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| P_{gs}^{comp}|] $',
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| GHZ\rangle\langle GHZ|] $', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| X^{\otimes rk}] $'], 
            'iterations': [iterations]
        }
        return schedule
    
    def linear_ala_gad_shift(self, perturbation, optimizer, iterations, target_locality=2):
        warnings.warn("Training schedule using Jordan and Fahri's gadget " + 
                      "construction. That construction has no guarantees of " +
                      "having the right low-energy spectrum for VQEs. " + 
                      "Use the new gadget construction instead.")
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        Hcomp = oH.computational()
        gadgetizer = PerturbativeGadgets(method='Jordan', 
                                         perturbation_factor=perturbation)
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality, offset_energy=True)
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = (r + 1) * int(k / (target_locality - 1))
        num_layers = self.n_comp + n_anc
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp + n_anc)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        schedule = {
            'name': 'linear_ala_gad_shift',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer], 
            'seed': self.np_rdm_seed,
            'ansaetze': [ala],
            'initial weights': initial_weights, 
            'training observables': [Hgad],
            'monitoring observables': [Hcomp, 
                                    #    oH.ancillary(), 
                                    #    oH.perturbation(), 
                                       Hgad, 
                                       oH.computational_ground_projector(), 
                                       gadgetizer.cat_projector(Hcomp, target_locality), 
                                       gadgetizer.ancillary_X(Hcomp, target_locality)],
            'labels': [r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| H^{comp}]$', 
                    #    r'$\langle \psi_{ALA}| H^{anc} |\psi_{ALA} \rangle$', 
                    #    r'$\langle \psi_{ALA}| \lambda V |\psi_{ALA} \rangle$', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| H^{gad}]$', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| P_{gs}^{comp}|] $', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| GHZ\rangle\langle GHZ|] $', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| X^{\otimes rk}] $'], 
            'iterations': [iterations]
        }
        return schedule
    
    def linear_ala_gad_penalized(self, perturbation, optimizer, iterations, 
                                 target_locality=2):
        warnings.warn("Training schedule using Jordan and Fahri's gadget " + 
                      "construction. That construction has no guarantees of " +
                      "having the right low-energy spectrum for VQEs. " + 
                      "Use the new gadget construction instead.")
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        Hcomp = oH.computational()
        gadgetizer = PerturbativeGadgets(method='Jordan', 
                                         perturbation_factor=perturbation)
        Htrain = gadgetizer.gadgetize(Hcomp, target_locality, 
                                    penalization=-1)
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality, 
                                    penalization=0)
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = r * int(k / (target_locality - 1))
        num_layers = self.n_comp + n_anc
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp + n_anc)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        schedule = {
            'name': 'linear_ala_gad_penalized',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer], 
            'seed': self.np_rdm_seed,
            'ansaetze': [ala],
            'initial weights': initial_weights, 
            'training observables': [Htrain],
            'monitoring observables': [Hcomp, 
                                    #    oH.ancillary(), 
                                    #    oH.perturbation(), 
                                       Hgad, 
                                       oH.computational_ground_projector(), 
                                       gadgetizer.cat_projector(Hcomp, target_locality), 
                                       gadgetizer.ancillary_X(Hcomp, target_locality)],
            'labels': [r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| H^{comp}]$', 
                    #    r'$\langle \psi_{ALA}| H^{anc} |\psi_{ALA} \rangle$', 
                    #    r'$\langle \psi_{ALA}| \lambda V |\psi_{ALA} \rangle$', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| H^{gad}]$', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| P_{gs}^{comp}|] $', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| GHZ\rangle\langle GHZ|] $', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| X^{\otimes rk}] $'], 
            'iterations': [iterations]
        }
        return schedule
    
    def linear_ala_gad_initialized(self, perturbation, optimizer, iterations, 
                                   target_locality=2):
        warnings.warn("Training schedule using Jordan and Fahri's gadget " + 
                      "construction. That construction has no guarantees of " +
                      "having the right low-energy spectrum for VQEs. " + 
                      "Use the new gadget construction instead.")
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        Hcomp = oH.computational()
        gadgetizer = PerturbativeGadgets(method='Jordan', 
                                         perturbation_factor=perturbation)
        Htrain = gadgetizer.gadgetize(Hcomp, target_locality, 
                                    penalization=-1)
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality)
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = r * int(k / (target_locality - 1))
        num_layers = self.n_comp + n_anc
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp + n_anc)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence, 
                                       initial_y_rot=False, 
                                       cat_range=range(self.n_comp, self.n_comp+n_anc), 
                                       undo_cat=False)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        # initial_weights *= 0.02
        initial_weights[:, self.n_comp:] *= 0.02
        schedule = {
            'name': 'linear_ala_gad_initialized',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer], 
            'seed': self.np_rdm_seed,
            'ansaetze': [ala],
            'initial weights': initial_weights, 
            'training observables': [Htrain],
            'monitoring observables': [Hcomp, 
                                    #    oH.ancillary(), 
                                    #    oH.perturbation(), 
                                       Hgad, 
                                       oH.computational_ground_projector(), 
                                       gadgetizer.cat_projector(Hcomp, target_locality), 
                                       gadgetizer.ancillary_X(Hcomp, target_locality)],
            'labels': [r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| H^{comp}]$', 
                    #    r'$\langle \psi_{ALA}| H^{anc} |\psi_{ALA} \rangle$', 
                    #    r'$\langle \psi_{ALA}| \lambda V |\psi_{ALA} \rangle$', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| H^{gad}]$', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| P_{gs}^{comp}|] $', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| GHZ\rangle\langle GHZ|] $', 
                       r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| X^{\otimes rk}] $'], 
            'iterations': [iterations]
        }
        return schedule
    
    def linear_ala_new_gad(self, perturbation, optimizer, iterations, 
                                 target_locality=3, initialize=False):
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        Hcomp = oH.computational()
        gadgetizer = NewPerturbativeGadgets(perturbation_factor=perturbation)
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality)
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = r * int(k / (target_locality - 2))
        num_layers = self.n_comp + n_anc
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp + n_anc)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        if initialize:
            initial_weights[:, self.n_comp:] *= 0.02
        schedule = {
            'name': 'linear_ala_new_gad',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer], 
            'seed': self.np_rdm_seed,
            'ansaetze': [ala],
            'initial weights': initial_weights, 
            'training observables': [Hgad],
            'monitoring observables': [Hcomp, 
                                       Hgad,  
                                       oH.computational_ground_projector(), 
                                       gadgetizer.zero_projector(Hcomp, target_locality), 
                                       gadgetizer.all_zero_projector(Hcomp, target_locality)],
            'labels': [r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| H^{comp}]$', 
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| H^{gad}]$', 
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| P_{gs}^{comp}|] $',
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| ' + 
                       r'\frac{1}{n_{anc}}\sum_{gad} | 0 \rangle\langle 0|] $',
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| ' + 
                       r'| 00\dots \rangle\langle 00\dots|] $'], 
            'iterations': [iterations]
        }
        return schedule
    
    def linear_ala_new_gad_lambda_scheduled(self, optimizer, iterations, 
                                            target_locality=3, 
                                            initialize=False):
        Hcomp = qml.Hamiltonian([1], [qml.operation.Tensor(*[qml.PauliZ(q) for q in range(self.n_comp)])])
        perturbations = [10, 3, 1, 0.3, 0.1]
        num_phases = len(perturbations)
        training_observables = []
        for perturbation in perturbations:
            gadgetizer = NewPerturbativeGadgets(perturbation_factor=perturbation)
            training_observables.append(gadgetizer.gadgetize(Hcomp, target_locality))
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = r * int(k / (target_locality - 2))
        num_layers = self.n_comp + n_anc
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp + n_anc)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        if initialize:
            initial_weights[:, self.n_comp:] *= 0.02
        schedule = {
            'name': 'linear_ala_new_gad_lambda_scheduled',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer] * num_phases, 
            'seed': self.np_rdm_seed,
            'ansaetze': [ala] * num_phases,
            'initial weights': initial_weights, 
            'training observables': training_observables,
            'monitoring observables': [Hcomp, 
                                    #    oH.computational_ground_projector(), 
                                       gadgetizer.zero_projector(Hcomp, target_locality), 
                                       gadgetizer.all_zero_projector(Hcomp, target_locality)],
            'labels': [r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| H^{comp}]$', 
                    #    r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| H^{gad}]$', 
                    #    r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| P_{gs}^{comp}|] $',
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| ' + 
                       r'\frac{1}{n_{anc}}\sum_{gad} | 0 \rangle\langle 0|] $',
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| ' + 
                       r'| 00\dots \rangle\langle 00\dots|] $'
                       ], 
            'iterations': [iterations] * num_phases
        }
        return schedule
    
    def linear_sala_new_gad(self, perturbation, optimizer, iterations, 
                                 target_locality=3, initialize=False):
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        Hcomp = oH.computational()
        gadgetizer = NewPerturbativeGadgets(perturbation_factor=perturbation)
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality)
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = r * int(k / (target_locality - 2))
        num_layers = self.n_comp + n_anc
        sala = SimplifiedAlternatingLayeredAnsatz(self.n_comp + n_anc, num_layers)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        if initialize:
            initial_weights[:, self.n_comp:] *= 0.02
        schedule = {
            'name': 'linear_sala_new_gad',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer], 
            'seed': self.np_rdm_seed,
            'ansaetze': [sala],
            'initial weights': initial_weights, 
            'training observables': [Hgad],
            'monitoring observables': [Hcomp, 
                                       Hgad,  
                                       oH.computational_ground_projector(), 
                                       gadgetizer.zero_projector(Hcomp, target_locality), 
                                       gadgetizer.all_zero_projector(Hcomp, target_locality)],
            'labels': [r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| H^{comp}]$', 
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| H^{gad}]$', 
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| P_{gs}^{comp}|] $',
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| ' + 
                       r'\frac{1}{n_{anc}}\sum_{gad} | 0 \rangle\langle 0|] $',
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| ' + 
                       r'| 00\dots \rangle\langle 00\dots|] $'], 
            'iterations': [iterations]
        }
        return schedule
    
    def linear_ala_reordered_gad(self, perturbation, optimizer, iterations, 
                                 target_locality=3, initialize=False):
        # oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        # Hcomp = oH.computational()
        terms = [qml.operation.Tensor(*[qml.PauliZ(q) for q in range(self.n_comp)])]
        Hcomp = qml.Hamiltonian([1], terms)
        gadgetizer = NewPerturbativeGadgets(perturbation_factor=perturbation)
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality)
        wires_map = gadgetizer.get_qubit_mapping(Hcomp, Hgad)
        reordered_Hgad = gadgetizer.map_wires(Hgad, wires_map)
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = r * int(k / (target_locality - 2))
        num_layers = self.n_comp + n_anc
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp + n_anc)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        if initialize:
            initial_weights[:, self.n_comp:] *= 0.02
        schedule = {
            'name': 'linear_ala_reordered_gad',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer], 
            'seed': self.np_rdm_seed,
            'ansaetze': [ala],
            'initial weights': initial_weights, 
            'training observables': [reordered_Hgad],
            'monitoring observables': [Hcomp, 
                                       Hgad,  
                                    #    oH.computational_ground_projector(), 
                                       gadgetizer.zero_projector(Hcomp, target_locality), 
                                       gadgetizer.all_zero_projector(Hcomp, target_locality)],
            'labels': [r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| H^{comp}]$', 
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| H^{gad}]$', 
                    #    r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| P_{gs}^{comp}|] $',
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| ' + 
                       r'\frac{1}{n_{anc}}\sum_{gad} | 0 \rangle\langle 0|] $',
                       r'$Tr[| \psi(\theta)\rangle\langle \psi(\theta)| ' + 
                       r'| 00\dots \rangle\langle 00\dots|] $'], 
            'iterations': [iterations]
        }
        reordered_monitoring_obs = []
        for H in schedule['monitoring observables']:
            reordered_monitoring_obs.append(gadgetizer.map_wires(H, wires_map))
        schedule['monitoring observables'] = reordered_monitoring_obs
        return schedule
    
    def linear_ala_comp(self, perturbation, optimizer, iterations):
        num_layers = self.n_comp
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp), 
                            requires_grad=True)
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        schedule = {
            'name': 'linear_ala_comp',
            'device': qml.device("default.qubit", 
            # 'device': qml.device("lightening.qubit", 
                                 wires=range(self.n_comp), 
                                 shots=self.num_shots),
            'optimizers': [optimizer],  
            'seed': self.np_rdm_seed,
            'ansaetze': [ala],
            'initial weights': initial_weights, 
            'training observables': [oH.computational()],
            'monitoring observables': [oH.computational(), 
                                       oH.computational_ground_projector()],
            'labels': [r'$\langle \psi_{ALA}| H^{comp} |\psi_{ALA} \rangle$',
                       r'$|\langle \psi_{ALA}| P_{gs}^{comp}| \psi_{ALA} \rangle |^2 $'], 
            'iterations': [iterations]
        }
        return schedule

    def shallow_ala_comp(self, perturbation, optimizer, iterations):
        num_layers = 2
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp), 
                            requires_grad=True)
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        schedule = {
            'name': 'shallow_ala_comp',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp), 
                                 shots=self.num_shots),
            'optimizers': [optimizer],  
            'seed': self.np_rdm_seed,
            'ansaetze': [ala],
            'initial weights': initial_weights, 
            'training observables': [oH.computational()],
            'monitoring observables': [oH.computational(), 
                                       oH.computational_ground_projector()],
            'labels': [r'$\langle \psi_{ALA}| H^{comp} |\psi_{ALA} \rangle$',
                       r'$|\langle \psi_{ALA}| P_{gs}^{comp}| \psi_{ALA} \rangle |^2 $'], 
            'iterations': [iterations]
        }
        return schedule

    def layerwise_ala_gad(self, perturbation, optimizer, iterations, 
                          layer_steps, target_locality=3):
        warnings.warn("Not tested, may have implementation bugs ")
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        Hcomp = oH.computational()
        gadgetizer = NewPerturbativeGadgets(perturbation_factor=perturbation)
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality)
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = r * int(k / (target_locality - 2))
        individual_depth = int(np.ceil(np.log(self.n_comp + n_anc)))
        num_layers = layer_steps * individual_depth
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp + n_anc)] 
                                for _ in range(num_layers)]
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        schedule = {
            'name': 'layerwise_ala_gad',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer] * layer_steps,  
            'seed': self.np_rdm_seed,
            'ansaetze': [AlternatingLayeredAnsatz(random_gate_sequence[:d*individual_depth]) 
                         for d in range(1, layer_steps+1, 1)],
            'initial weights': initial_weights, 
            'training observables': [Hgad] * layer_steps,
            'monitoring observables': [Hcomp, 
                                       Hgad, 
                                       oH.computational_ground_projector()],
            'labels': [r'$\langle \psi_{ALA}| H^{comp} |\psi_{ALA} \rangle$', 
                       r'$\langle \psi_{ALA}| H^{gad} |\psi_{ALA} \rangle$', 
                       r'$|\langle \psi_{ALA}| P_{gs}^{comp}| \psi_{ALA} \rangle |^2 $'], 
            'iterations': [iterations] * layer_steps
        }
        return schedule
    
    def ala_gad_to_comp(self, perturbation, optimizer, iterations, target_locality=3):
        warnings.warn("Not tested, may have implementation bugs ")
        oH = ObservablesHolmes(self.n_comp, 0, perturbation)
        Hcomp = oH.computational()
        gadgetizer = NewPerturbativeGadgets(perturbation_factor=perturbation)
        Hgad = gadgetizer.gadgetize(Hcomp, target_locality)
        _, k, r = gadgetizer.get_params(Hcomp)
        n_anc = r * int(k / (target_locality - 2))
        num_layers = self.n_comp + n_anc
        random_gate_sequence = [[np.random.choice(self.gate_set) 
                                for _ in range(self.n_comp + n_anc)] 
                                for _ in range(num_layers)]
        ala = AlternatingLayeredAnsatz(random_gate_sequence)
        initial_weights = np.random.uniform(0, np.pi, 
                            size=(num_layers, self.n_comp + n_anc), 
                            requires_grad=True)
        schedule = {
            'name': 'ala_gad_to_comp',
            'device': qml.device("default.qubit", 
                                 wires=range(self.n_comp + n_anc), 
                                 shots=self.num_shots),
            'optimizers': [optimizer] * 2,  
            'seed': self.np_rdm_seed,
            'ansaetze': [ala] * 2,
            'initial weights': initial_weights, 
            'training observables': [Hgad, Hcomp],
            'monitoring observables': [Hcomp, 
                                       Hgad, 
                                       oH.computational_ground_projector()],
            'labels': [r'$\langle \psi_{ALA}| H^{comp} |\psi_{ALA} \rangle$', 
                       r'$\langle \psi_{ALA}| H^{gad} |\psi_{ALA} \rangle$', 
                       r'$|\langle \psi_{ALA}| P_{gs}^{comp}| \psi_{ALA} \rangle |^2 $'], 
            'iterations': [iterations] * 2
        }
        return schedule



