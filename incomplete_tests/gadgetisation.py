from logging import raiseExceptions
import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np


target_locality = 2
perturbation_factor = 1

target_coeffs = [0.5, 0.5]
target_obs = [qml.PauliZ(0) @ qml.PauliX(1) @ qml.PauliY(2), 
              qml.PauliZ(1) @ qml.PauliX(2) @ qml.PauliY(3)]
Hcomp = qml.Hamiltonian(target_coeffs, target_obs)

print("Studied Hamiltonian: \n", Hcomp)
print("Number of terms:     ", len(Hcomp.ops))
print("Acting on qubits:    ", Hcomp.wires)
computational_qubits = len(Hcomp.wires)
total_qubits = computational_qubits
if computational_qubits != Hcomp.wires[-1]+1:
    raise Exception('The studied computational Hamiltonian is not acting on ' + 
                    'the first {} qubits. '.format(computational_qubits) + 
                    'Decomposition not implemented for this case')
# Check for same string lengths
localities=[]
for string in Hcomp.ops:
    localities.append(len(string.non_identity_obs))
if len(np.unique(localities)) > 1:
    raise Exception('The given Hamiltonian has terms with different locality.' +
                    ' Gadgetization not implemented for this case')
else: 
    computational_locality = localities[0]
l = perturbation_factor * (computational_locality - 1) / (4 * computational_locality)
print("creating the gadget Hamiltonian")
coeffs_anc = []
coeffs_pert = []
obs_anc = []
obs_pert = []
for str_count, string in enumerate(Hcomp.ops):
    print("Term ", str_count+1, ":            ", Hcomp.coeffs[str_count], string)
    ancillary_qubits = int(len(string.non_identity_obs) / (target_locality - 1))
    previous_total = total_qubits
    total_qubits += ancillary_qubits
    # print("Required number of ancillary qubits: ", ancillary_qubits)
    for first in range(previous_total, total_qubits):
        for second in range(first+1, total_qubits):
            coeffs_anc += [0.5, -0.5]
            obs_anc += [qml.Identity(first) @ qml.Identity(second), 
                    qml.PauliZ(first) @ qml.PauliZ(second)]
    for anc_q in range(ancillary_qubits):
        term = qml.PauliX(previous_total+anc_q)
        term = qml.operation.Tensor(term, *string.non_identity_obs[
                (target_locality-1)*anc_q:(target_locality-1)*(anc_q+1)])
        obs_pert.append(term)
        print(term)
    coeffs_pert += [l] * ancillary_qubits
Hanc = qml.Hamiltonian(coeffs_anc, obs_anc)
Hpert = qml.Hamiltonian(coeffs_pert, obs_pert)
print(Hanc)
print(Hpert)
