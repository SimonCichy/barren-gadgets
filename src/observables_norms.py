import pennylane as qml
from pennylane import numpy as np

from observables_holmes import ObservablesHolmes

np.random.seed(42)
data_folder = '../results/data/norms/'

def generate_norms(observable, qubit_range, lambda_scaling):
    file_name = data_folder + observable + '_{}qubits_{}lambda.dat'.format(qubit_range[-1], lambda_scaling)
    with open(file_name, 'w') as of:
        of.write('# qubits\tinf norm\tone norm\n')
    inf_norm = []
    coeffs_norm = []
    for n in qubit_range:
        if observable == 'global':
            oH = ObservablesHolmes(n, 0, lambda_scaling)
            obs = oH.computational()
        elif observable == 'local':
            oH = ObservablesHolmes(n, 0, lambda_scaling)
            obs = oH.local()
        elif observable == 'gadget':
            oH = ObservablesHolmes(n, n, lambda_scaling)
            obs = oH.gadget()
        inf_norm.append(np.linalg.norm(qml.matrix(obs), ord=np.inf))
        coeffs_norm.append(np.linalg.norm(obs.coeffs, ord=1))
        with open(file_name, 'a') as of:
            of.write('{}\t{}\t{}\n'.format(n, inf_norm[-1], coeffs_norm[-1]))

if __name__ == "__main__":
    num_qubits = np.arange(2, 12, 2)
    observable = "gadget"
    lambda_scaling = 0.5
    generate_norms(observable, num_qubits, lambda_scaling)
