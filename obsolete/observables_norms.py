import pennylane as qml
from pennylane import numpy as np

from observables_holmes import ObservablesHolmes

np.random.seed(42)
data_folder = '../results/data/'

def generate_norms(observable, qubit_range, lambda_range):
    file_name = data_folder + 'norms/' + observable + \
                '_{}-{}qubits_{}-{}lambda.dat'.format(qubit_range[0], qubit_range[-1], lambda_range[0], lambda_range[-1])
    with open(file_name, 'w') as of:
        of.write('# qubits\tperturbation\tinf norm\tone norm\n')
    inf_norm = []
    coeffs_norm = []
    for n in qubit_range:
        for l in lambda_range:
            if observable == 'global':
                oH = ObservablesHolmes(n, 0, l)
                obs = oH.computational()
            elif observable == 'local':
                oH = ObservablesHolmes(n, 0, l)
                obs = oH.local()
            elif observable == 'gadget':
                oH = ObservablesHolmes(n, n, l)
                obs = oH.gadget()
            inf_norm.append(np.linalg.norm(qml.matrix(obs), ord=np.inf))
            coeffs_norm.append(np.linalg.norm(obs.coeffs, ord=1))
            with open(file_name, 'a') as of:
                of.write('{}\t{:1.1f}\t{}\t{}\n'.format(n, l, inf_norm[-1], coeffs_norm[-1]))

def generate_ground_energies(observable, qubit_range, lambda_range):
    file_name = data_folder + 'energies/' + observable + \
                '_{}-{}qubits_{}-{}lambda.dat'.format(qubit_range[0], qubit_range[-1], lambda_range[0], lambda_range[-1])
    with open(file_name, 'w') as of:
        of.write('# qubits\tperturbation\tground_energy\n')
    ge = []
    for n in qubit_range:
        for l in lambda_range:
            if observable == 'global':
                oH = ObservablesHolmes(n, 0, l)
                obs = oH.computational()
            elif observable == 'local':
                oH = ObservablesHolmes(n, 0, l)
                obs = oH.local()
            elif observable == 'gadget':
                oH = ObservablesHolmes(n, n, l)
                obs = oH.gadget()
            ge.append(min(qml.eigvals(obs)))
            with open(file_name, 'a') as of:
                of.write('{}\t{:1.1f}\t{}\n'.format(n, l, ge[-1]))

if __name__ == "__main__":
    # num_qubits = np.arange(2, 12, 2)
    num_qubits = [4]
    observable = "gadget"
    lambda_range = np.linspace(0, 1, 6)
    # generate_norms(observable, num_qubits, lambda_range)
    generate_ground_energies(observable, num_qubits, lambda_range)
