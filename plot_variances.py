import numpy as np
import matplotlib.pyplot as plt

colours = ['#6f45d8', '#4547d8', '#45ccd8', '#45d859', '#c2d845', '#d89d45', '#d85645']

file = '../results/data/global_circuit_4qubits_5layers_200samples.dat'

data = np.loadtxt(file)

layers = data[:,0].astype(int)
qubits = data[:,1].astype(int)
gradients = data[:,2:]

# getting the different numbers of layers tested
layers_list = np.unique(layers)
for nl, num_layers in enumerate(layers_list):
    # getting the indexes of the corresponding rows
    row_indexes = np.where(layers==num_layers)
    # selecting the corresponding rows of the data
    grad_vals = gradients[row_indexes]
    qubits_list = qubits[row_indexes]
    variance_vals = np.var(grad_vals, axis=1)
    
    plt.semilogy(qubits_list, variance_vals, "--o", c=colours[nl], label="{} layers".format(num_layers))

plt.xlabel(r"N Qubits")
plt.ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
plt.legend()
plt.show()