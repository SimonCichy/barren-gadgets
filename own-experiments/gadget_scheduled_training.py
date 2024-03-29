import sys
sys.path.append('../src')
sys.path.append('src')
import time
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

from trainings import scheduled_training, SchedulesOfInterest

# data_folder = '../results/data/training/'
use_exact_ground_energy = False
plot_data = True
save_data = True

computational_qubits = 5
newk = 3
max_iter = 500
step = 0.3
num_shots = None

opt = qml.GradientDescentOptimizer(stepsize=step)

def repeated_training():
    seeds = range(1, 31)
    # perturbation_factors = np.linspace(0, 1, 6)
    perturbation_factors = [0.1, 1, 10] 
    for seed in seeds:
        np.random.seed(seed)
        tic = time.perf_counter()
        for pf in perturbation_factors:
            print(" Computational qubits:   ", computational_qubits)
            print(" Random seed:            ", seed)
            print(" Perturbation factor:    ", pf)
            soi = SchedulesOfInterest(computational_qubits, 
                                      seed, num_shots)
            schedule = soi.linear_ala_reordered_gad(pf, opt, max_iter, newk, False)
            scheduled_training(schedule, plot_data=plot_data, save_data=save_data)

        toc = time.perf_counter()
        print(f"Script run in {toc - tic:0.4f} seconds")
        if plot_data:
            plt.show()

def decreasing_lambda():
    seed = 1
    np.random.seed(seed)
    tic = time.perf_counter()
    print(" Computational qubits:   ", computational_qubits)
    print(" Random seed:            ", seed)
    soi = SchedulesOfInterest(computational_qubits, 
                                seed, num_shots)
    schedule = soi.linear_ala_new_gad_lambda_scheduled(opt, max_iter, newk, False)
    scheduled_training(schedule, plot_data=plot_data, save_data=save_data)

    toc = time.perf_counter()
    print(f"Script run in {toc - tic:0.4f} seconds")
    if plot_data:
        plt.show()


if __name__ == "__main__":
    repeated_training()
    # decreasing_lambda()
