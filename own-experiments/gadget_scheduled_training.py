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

computational_qubits = 4
max_iter = 100
step = 0.3
num_shots = None

# perturbation_factors = np.linspace(0, 1, 6)
perturbation_factors = [1]
opt = qml.GradientDescentOptimizer(stepsize=step)


if __name__ == "__main__":
    print(" Computational qubits:   ", computational_qubits)
    # print(" Ancillary qubits:       ", ancillary_qubits)
    print(" Random seed:            ", seed)
    tic = time.perf_counter()
    for pf in perturbation_factors:
        print(" Perturbation factor:    ", pf)
        soi = SchedulesOfInterest(computational_qubits, 
                                  seed, num_shots)
        schedule = soi.shallow_ala_gad(pf, opt, max_iter, 3)
        scheduled_training(schedule, plot_data=plot_data, save_data=save_data)

    toc = time.perf_counter()
    print(f"Script run in {toc - tic:0.4f} seconds")
    plt.show()
