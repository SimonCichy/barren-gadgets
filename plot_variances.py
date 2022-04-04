import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

colours = ['#6f45d8', '#4547d8', '#45ccd8', '#45d859', '#c2d845', '#d89d45', '#d85645']

file1 = '../results/data/220401_local_circuit_6qubits_50layers_1000samples.dat'
file2 = '../results/data/220402_gadget_circuit_6qubits_50layers_1000samples.dat'

data_local = np.loadtxt(file1)
data_gadget = np.loadtxt(file2)

layers = data_local[:,0].astype(int)
qubits = data_local[:,1].astype(int)
gradients_local = data_local[:,2:]
gradients_gadget = data_gadget[:,2:]

fig, ax = plt.subplots()
ax2 = ax.twiny() 

# getting the different numbers of layers tested
layers_list = np.unique(layers)
for nl, num_layers in enumerate(layers_list):
    # getting the indexes of the corresponding rows
    row_indexes = np.where(layers==num_layers)
    # selecting the corresponding rows of the data
    qubits_list = qubits[row_indexes]
    grad_vals_loc = gradients_local[row_indexes]
    grad_vals_gad = gradients_gadget[row_indexes]
    variance_vals_loc = np.var(grad_vals_loc, axis=1)
    variance_vals_gad = np.var(grad_vals_gad, axis=1)
    
    ax.semilogy(qubits_list, variance_vals_loc, "--o", c=colours[nl], label="{} layers".format(num_layers))
    ax2.semilogy(2*qubits_list, variance_vals_gad, ":v", c=colours[nl], alpha=0.7)

ax.set_ylim([2.5e-3, 2.2e-1])
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