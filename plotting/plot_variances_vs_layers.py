import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

colours = [['#36226b', '#22236b', '#22656b', '#226b2b', '#606b22', '#6b4d22', '#6b2a22'],
           ['#6f45d8', '#4547d8', '#45ccd8', '#45d859', '#c2d845', '#d89d45', '#d85645'],
           ['#977dd8', '#7d7fd8', '#7dd1d8', '#7dd889', '#cbd87d', '#d8b47d', '#d8887d']]

file_gad = '../results/data/220404_gadget_circuit_2qubits_50layers_200samples.dat'


data_gadget = np.loadtxt(file_gad)

layers = data_gadget[:,0].astype(int)
lamb = data_gadget[:,1]
gradients_gadget = data_gadget[:,2:]

fig, ax = plt.subplots()
# ax2 = ax.twiny() 

# getting the different numbers of layers tested
lambdas_list = np.unique(lamb)
for l, lam in enumerate(lambdas_list):
    # getting the indexes of the corresponding rows
    row_indexes = np.where(lamb==lam)
    # selecting the corresponding rows of the data
    layers_list = layers[row_indexes]
    grad_vals_gad = gradients_gadget[row_indexes]
    variance_vals_gad = np.var(grad_vals_gad, axis=1)
    # rescaling factor
    # k = qubits_list
    # norm_gadget = r * k*(k-1) + lam * (1 + k - 1)
    norm_gadget = 1
    
    ax.semilogy(layers_list, variance_vals_gad/norm_gadget**2, "-v", c=colours[1][l], label=r"$\lambda=${}".format(lam))

# ax.set_ylim([2.5e-3, 2.2e-1])
ax.set_xlabel(r"Number of layers")
ax.set_ylabel(r"$\langle \partial \theta_{1, 1} E\rangle$ variance")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.legend()
ax.set_title(r"$n_{comp}=2$")

plt.show()