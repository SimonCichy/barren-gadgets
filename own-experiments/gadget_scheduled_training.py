import sys
sys.path.append('../src')
sys.path.append('src')
import time
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from trainings import scheduled_training, SchedulesOfInterest

seed = 2
np.random.seed(seed)
data_folder = '../results/data/training/'
use_exact_ground_energy = False
plot_data = True
save_data = False

computational_qubits = 10
ancillary_qubits = int(1 * computational_qubits)
max_iter = 100
step = 0.3
num_shots = None

# perturbation_factors = np.linspace(0, 1, 6)
perturbation_factors = [1]

dev_comp = qml.device("default.qubit", 
# dev_comp = qml.device("lightning.qubit", 
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
    tic = time.perf_counter()
    for pf in perturbation_factors:
        print(" Perturbation factor:    ", pf)
        soi = SchedulesOfInterest(computational_qubits, ancillary_qubits, 
                                  dev_comp, dev_gad, seed)
        schedule = soi.shallow_ala_comp(pf, opt, max_iter)
        scheduled_training(schedule, plot_data=plot_data, save_data=save_data)

    toc = time.perf_counter()
    print(f"Script run in {toc - tic:0.4f} seconds")
    plt.show()
