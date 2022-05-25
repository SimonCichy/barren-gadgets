import sys
sys.path.append('../src')
sys.path.append('src')
import math
import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

from jordan_gadgets import PerturbativeGadgets

# Generating test Hamiltonians:
H1 = qml.Hamiltonian([1], [qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2)])
H2 = qml.Hamiltonian([1], [qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2) @ qml.PauliZ(3)])
H3 = qml.Hamiltonian([1, 1], [qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliZ(2), 
                              qml.PauliX(0) @ qml.PauliY(1) @ qml.PauliY(2)])

# Generating the corresponding gadgetized gadget Hamiltonians: 
gadgetizer = PerturbativeGadgets(method='Jordan', perturbation_factor=1)
H1gad = gadgetizer.gadgetize(H1, target_locality=2)
H2gad = gadgetizer.gadgetize(H2, target_locality=2)
H3gad = gadgetizer.gadgetize(H3, target_locality=2)

def get_ideal_effective(Hamiltonian, gadgetizer):
    # getting the locality, assuming all terms have the same
    _, k, r = gadgetizer.get_params(Hamiltonian)
    # calculation the perturbation factor lambda
    l = gadgetizer.perturbation_factor * (k - 1) / (4 * k)
    factor = - k * (-l)**k / math.factorial(k-1)
    # implementing the projector on the GHZ state
    Hid = factor * qml.matrix(Hamiltonian)
    projector_list = gadgetizer.cat_projector(Hamiltonian).ops
    assert len(projector_list) == r
    for Pplus in projector_list:
        Hid = np.kron(Hid, qml.matrix(Pplus))
    return Hid

# Generating the corresponding ideal effective Hamiltonian:
H1id = get_ideal_effective(H1, gadgetizer)
H2id = get_ideal_effective(H2, gadgetizer)
H3id = get_ideal_effective(H3, gadgetizer)

def get_norm_error(H, l):
    # dimension of thhe target Hamiltonian (2^n)
    n = len(H.wires)
    dim = 2^n
    # creating the gadgetizer object
    gadgetizer = PerturbativeGadgets(method='Jordan', perturbation_factor=l)
    # getting the locality of the target Hamiltonian
    _, k, _ = gadgetizer.get_params(H)
    # creating the corresponding ideal Hamiltonian on computational and ancillary registers
    Hid = get_ideal_effective(H, gadgetizer)
    # computing the norm of the ideal Hamiltonian
    norm_id = np.linalg.norm(Hid, ord=np.inf)
    # generating the corresponding target Hamiltonian to the target Hamiltonian
    Hgad = qml.matrix(gadgetizer.gadgetize(H, target_locality=2))
    # removing the highest energy eigenvalues
    Eigenvalues, Eigenvectors = np.linalg.eigh(np.array(Hgad))
    for _ in range(dim, len(Eigenvalues)):
        Eigenvalues[np.argmax(Eigenvalues)] = 0
    Lambda = np.diag(Eigenvalues)
    Heff = Eigenvectors @ Lambda @ Eigenvectors.T
    # creating the difference Hamiltonian and getting the norm
    Hdiff = Hid - Heff
    norm_diff = np.linalg.norm(Hdiff, ord=np.inf)
    return norm_diff/norm_id, l * (k - 1) / (4 * k)



if __name__ == "__main__":
    print("Gadgetization: ")
    print('H1 = ', H1)
    print(qml.matrix(H1))
    # print(np.linalg.eig(H1id))
    print('Hgad(H1) = ', H1gad)
    errors_1 = []
    errors_2 = []
    errors_3 = []
    lambda_values_1 = []
    lambda_values_2 = []
    lambda_values_3 = []
    plot_points = 10
    step = 1/plot_points
    lambda_range = np.arange(step, 1+step, step)
    for lamb in lambda_range:
        err, l = get_norm_error(H1, lamb)
        errors_1.append(err)
        lambda_values_1.append(l)
        err, l = get_norm_error(H2, lamb)
        errors_2.append(err)
        lambda_values_2.append(l)
        err, l = get_norm_error(H3, lamb)
        errors_3.append(err)
        lambda_values_3.append(l)
    plt.plot(lambda_values_1, errors_1, '^', mec='k', mfc='none')
    plt.plot(lambda_values_2, errors_2, 's', mec='k', mfc='none')
    plt.plot(lambda_values_3, errors_3, 'o', mec='k', mfc='none')
    # plt.ylim([0, 0.25])
    plt.show()
    print(np.linalg.norm(H1id, ord=np.inf))
    print(np.linalg.norm(qml.matrix(H1gad), ord=np.inf))
    # print('H2 = ', H2)
    # print('Hgad(H2) = ', H2gad)
    # print('H3 = ', H3)
    # print('Hgat(H3) = ', H3gad)
    


