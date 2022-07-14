import sys
sys.path.append('../src')
sys.path.append('src')
import numpy as np
from data_management import save_gradients

data_folder = '../results/data/'
file1 = data_folder + 'gradients/220712_euler/gradients_nr0002.npz'
file2 = data_folder + 'gradients/220713_euler/gradients_nr0001.npz'

data1 = np.load(file1, allow_pickle=True)
data2 = np.load(file2, allow_pickle=True)

comp_qubits1 = data1['computational_qubits']
comp_qubits2 = data2['computational_qubits']
layers1 = data1['layers_list']
layers2 = data2['layers_list']
total_qubits1 = data1['widths_list']
total_qubits2 = data2['widths_list']
variances1 = data1['variances_list']
variances2 = data2['variances_list']
norms1 = data1['norms_list']
norms2 = data2['norms_list']
gradients1 = data1['all_gradients'].item()
gradients2 = data2['all_gradients'].item()

# trimming the list of computational qubits 
# to the list of actually simulated ones
comp_qubits1 = comp_qubits1[:len(total_qubits1)]
comp_qubits2 = comp_qubits2[:len(total_qubits2)]

comp_qubitst = np.append(comp_qubits1, comp_qubits2)
layerst = layers1
total_qubitst = np.append(total_qubits1, total_qubits2) 
variancest = np.append(variances1, variances2, axis=1) 
normst = np.append(norms1, norms2) 
gradientst = {}
for key in gradients1.keys():
    gradientst[key] = gradients1[key]
for key in gradients2.keys():
    gradientst[key] = gradients2[key]

# print(comp_qubitst)
# print(layerst)
# print(total_qubitst)
# print(variancest)
# print(normst)
# print(gradientst.keys())

data_dict = {
    'computational qubits': comp_qubitst,
    'layers': layerst,
    'widths': total_qubitst,
    'norms': normst,
    'variances': variancest, 
    'all gradients': gradientst
}
save_gradients(data_dict, perturbation_factor=1, mode='new file')