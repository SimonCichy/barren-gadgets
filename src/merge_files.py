from statistics import variance
import sys
sys.path.append('../src')
sys.path.append('src')
import numpy as np
from data_management import save_gradients

data_folder = '../results/data/'
file1 = data_folder + 'gradients/220811/gradients_nr0001.npz'
file2 = data_folder + 'gradients/220819/gradients_nr0001.npz'

data1 = np.load(file1, allow_pickle=True)
data2 = np.load(file2, allow_pickle=True)

# def check_shapes():
#     print(gradients1.keys())
#     print(gradients2.keys())
#     print(np.shape(gradients1[(8, 2)]))
#     print(np.shape(gradients2[(8, 2)]))
#     # print(np.shape(np.append(gradients1[(12, 2)], gradients2[(12, 2)], axis=0)))

def merge_in_width():

    comp_qubits = {
        1 : data1['computational_qubits'], 
        2 : data2['computational_qubits']}
    layers = {
        1 : data1['layers_list'],
        2 : data2['layers_list']}
    total_qubits = {
        1 : data1['widths_list'],
        2 : data2['widths_list']}
    variances = {
        1 : data1['variances_list'],
        2 : data2['variances_list']}
    norms = {
        1 : data1['norms_list'],
        2 : data2['norms_list']}
    gradients = {
        1 : data1['all_gradients'].item(),
        2 : data2['all_gradients'].item()}

    # trimming the list of computational qubits 
    # to the list of actually simulated ones
    comp_qubits[1] = comp_qubits[1][:len(total_qubits[1])]
    comp_qubits[2] = comp_qubits[2][:len(total_qubits[2])]

    comp_qubits['t'] = np.append(comp_qubits[1], comp_qubits[2])
    comp_qubits['t'] = np.sort(comp_qubits['t'])
    layers['t'] = layers[1]
    total_qubits['t'] = []
    variances['t'] = [0 * variances[1][0]]
    norms['t'] = []
    gradients['t'] = {}
    for n_comp in comp_qubits['t']:
        if n_comp in comp_qubits[1]:
            if n_comp in comp_qubits[2]:
                print("WARNING: dupplicate!")
            source = int(1)
        elif n_comp in comp_qubits[2]:
            source = int(2)
        index = np.where(comp_qubits[source] == n_comp)
        total_qubits['t'] = np.append(total_qubits['t'], total_qubits[source][index]) 
        variances['t'] = np.append(variances['t'], variances[source][index], axis=1) 
        norms['t'] = np.append(norms['t'], norms[source][index])
        for key in gradients[source].keys():
            if key[0] == n_comp:
                gradients['t'][key] = gradients[source][key]
    variances['t'] = variances['t'][1:]

    data_dict = {
        'computational qubits': comp_qubits['t'],
        'layers': layers['t'],
        'widths': total_qubits['t'],
        'norms': norms['t'],
        'variances': variances['t'], 
        'all gradients': gradients['t']
    }
    save_gradients(data_dict, perturbation_factor=1, mode='new file')

def merge_in_samples():

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

    assert (comp_qubits1 == comp_qubits2).all
    assert (layers1 == layers2).all
    assert (total_qubits1 == total_qubits2).all
    assert (norms1 == norms2).all
    assert gradients1.keys() == gradients2.keys()
    gradientst = {}
    for key in gradients1.keys():
        gradientst[key] = np.append(gradients1[key], gradients2[key], axis=0)

    data_dict = {
        'computational qubits': comp_qubits1,
        'layers': layers1,
        'widths': total_qubits1,
        'norms': norms1,
        'variances': None, 
        'all gradients': gradientst
    }
    save_gradients(data_dict, perturbation_factor=1, mode='new file')


if __name__ == "__main__":
    # check_shapes()
    merge_in_width()
    # merge_in_samples()