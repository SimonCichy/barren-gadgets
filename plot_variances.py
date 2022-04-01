import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

colours = ['#6f45d8', '#4547d8', '#45ccd8', '#45d859', '#c2d845', '#d89d45', '#d85645']

file = '../results/data/220401_global_circuit_6qubits_50layers_1000samples.dat'

data = np.loadtxt(file)

layers = data[:,0].astype(int)
qubits = data[:,1].astype(int)
gradients = data[:,2:]

fig, ax = plt.subplots()

# getting the different numbers of layers tested
layers_list = np.unique(layers)
for nl, num_layers in enumerate(layers_list):
    # getting the indexes of the corresponding rows
    row_indexes = np.where(layers==num_layers)
    # selecting the corresponding rows of the data
    grad_vals = gradients[row_indexes]
    qubits_list = qubits[row_indexes]
    variance_vals = np.var(grad_vals, axis=1)
    
    ax.semilogy(qubits_list, variance_vals, "--o", c=colours[nl], label="{} layers".format(num_layers))

# ax.set_ylim([2.5e-3, 1.25e-1])
ax.set_xlabel(r"N Qubits")
ax.set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend()
plt.show()