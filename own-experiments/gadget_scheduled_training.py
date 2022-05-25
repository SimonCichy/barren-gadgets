import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from hardware_efficient_ansatz import AlternatingLayeredAnsatz
from observables_holmes import ObservablesHolmes
from jordan_gadgets import PerturbativeGadgets
from trainings import scheduled_training, SchedulesOfInterest

seed = 4
np.random.seed(seed)
data_folder = '../results/data/training/'
use_exact_ground_energy = False
plot_data = True
save_data = False

computational_qubits = 8
ancillary_qubits = int(1 * computational_qubits)
# num_layers = computational_qubits + ancillary_qubits
max_iter = 100
step = 0.3
num_shots = None

# gate_set = [qml.RX, qml.RY, qml.RZ]
# perturbation_factors = np.linspace(0, 1, 6)
perturbation_factors = [1]

dev_comp = qml.device("default.qubit", 
                      wires=range(computational_qubits), 
                      shots=num_shots)
dev_gad = qml.device("default.qubit", 
                     wires=range(computational_qubits+ancillary_qubits), 
                     shots=num_shots)
opt = qml.GradientDescentOptimizer(stepsize=step)


if __name__ == "__main__":
    print(" Computational qubits:   ", computational_qubits)
    print(" Ancillary qubits:       ", ancillary_qubits)
    print(" Random seed:            ", seed)
    for pf in perturbation_factors:
        print(" Perturbation factor:    ", pf)
        # oH = ObservablesHolmes(computational_qubits, ancillary_qubits, pf)
        # gadgetizer = PerturbativeGadgets(method='Jordan', perturbation_factor=pf)
        # random_gate_sequence = [[np.random.choice(gate_set) 
        #                         for _ in range(computational_qubits+ancillary_qubits)] 
        #                         for _ in range(num_layers)]
        # ala1 = AlternatingLayeredAnsatz(random_gate_sequence[:-ancillary_qubits])
        # ala2 = AlternatingLayeredAnsatz(random_gate_sequence)
        # initial_weights = np.random.uniform(0, np.pi, 
        #                     size=(num_layers, computational_qubits+ancillary_qubits), 
        #                     requires_grad=True)
        # schedule = {
        #     'device': dev_gad,
        #     'optimizers': [opt] * 2, 
        #     'ansaetze': [ala1, ala2],
        #     'initial weights': initial_weights[:-ancillary_qubits], 
        #     'training observables': [oH.gadget()] * 2,
        #     'monitoring observables': [oH.computational(), 
        #                                oH.ancillary(), 
        #                                oH.perturbation(), 
        #                                oH.gadget(), 
        #                                gadgetizer.cat_projector(), 
        #                                oH.computational_ground_projector()],
        #     'labels': [r'$\langle \psi_{ALA}| H^{comp} |\psi_{ALA} \rangle$', 
        #                r'$\langle \psi_{ALA}| H^{anc} |\psi_{ALA} \rangle$', 
        #                r'$\langle \psi_{ALA}| \lambda V |\psi_{ALA} \rangle$', 
        #                r'$\langle \psi_{ALA}| H^{gad} |\psi_{ALA} \rangle$', 
        #                r'$Tr[| \psi_{ALA}\rangle\langle \psi_{ALA}| GHZ\rangle\langle GHZ|] $', 
        #                r'$|\langle \psi_{ALA}| P_{gs}^{comp}| \psi_{ALA} \rangle |^2 $'], 
        #     'iterations': [max_iter, 0.5*max_iter]
        # }

        soi = SchedulesOfInterest(computational_qubits, ancillary_qubits, 
                                  dev_comp, dev_gad)
        schedule = soi.linear_ala_comp(pf, opt, max_iter)
        scheduled_training(schedule, plot_data=plot_data, save_data=save_data)

    plt.show()
