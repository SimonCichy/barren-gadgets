import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

colours = [['#36226b', '#22236b', '#22656b', '#226b2b', '#606b22', '#6b4d22', '#6b2a22'],
           ['#6f45d8', '#4547d8', '#45ccd8', '#45d859', '#c2d845', '#d89d45', '#d85645'],
           ['#977dd8', '#7d7fd8', '#7dd1d8', '#7dd889', '#cbd87d', '#d8b47d', '#d8887d']]

file_glob = '../results/data/220401_global_circuit_6qubits_50layers_1000samples.dat'
file_loc = '../results/data/220401_local_circuit_6qubits_50layers_1000samples.dat'
file_gad = '../results/data/220402_gadget_circuit_6qubits_50layers_1000samples.dat'

data_global = np.loadtxt(file_glob)
data_local = np.loadtxt(file_loc)
data_gadget = np.loadtxt(file_glob)

layers = data_local[:,0].astype(int)
qubits = data_local[:,1].astype(int)
gradients_global = data_global[:,2:]
gradients_local = data_local[:,2:]
gradients_gadget = data_gadget[:,2:]

fig, ax = plt.subplots()
ax2 = ax.twiny() 

r = 1
lam = 1

# getting the different numbers of layers tested
layers_list = np.unique(layers)
for nl, num_layers in enumerate(layers_list):
    # getting the indexes of the corresponding rows
    row_indexes = np.where(layers==num_layers)
    # selecting the corresponding rows of the data
    qubits_list = qubits[row_indexes]
    grad_vals_glob = gradients_global[row_indexes]
    grad_vals_loc = gradients_local[row_indexes]
    grad_vals_gad = gradients_gadget[row_indexes]
    variance_vals_glob = np.var(grad_vals_glob, axis=1)
    variance_vals_loc = np.var(grad_vals_loc, axis=1)
    variance_vals_gad = np.var(grad_vals_gad, axis=1)
    # rescaling factor
    k = qubits_list
    norm_global = r
    norm_local = 1
    norm_gadget = r * k*(k-1) + lam * (1 + k - 1)
    
    ax.semilogy(qubits_list, variance_vals_glob/norm_global**2, "--s", c=colours[0][nl])
    ax2.semilogy(2*qubits_list, variance_vals_gad/norm_gadget**2, "-v", c=colours[1][nl], label="{} layers".format(num_layers))
    ax.semilogy(qubits_list, variance_vals_loc/norm_local**2, ":o", c=colours[2][nl])

# ax.set_ylim([2.5e-3, 2.2e-1])
ax.set_xlabel(r"N Computational Qubits")
ax.set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.set_xlabel(r"N Total Qubits", color = 'grey') 
ax2.tick_params(axis ='x', labelcolor = 'grey')
ax2.xaxis.set_major_locator(MultipleLocator(2))
# check for common legend: 
# https://stackoverflow.com/questions/48391146/change-marker-in-the-legend-in-matplotlib
ax.legend()

plt.show()