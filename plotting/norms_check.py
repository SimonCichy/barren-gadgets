import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

from src.observables_holmes import ObservablesHolmes

num_qubits = np.arange(2, 7)            # numbber of computational qubits
k = num_qubits                          # locality
r = 1                                   # number of terms
lam_scale = 0.5                         # variational parameter λ
lam = lam_scale * (num_qubits - 1) / (4 * num_qubits)

upper_bounds = {}
for locality in [2]:
    ktilde = k / (locality - 1)
    upper_bounds[locality] = 0.5 * r * ktilde * (ktilde - 1) + r * lam * ktilde


one_norm = []
inf_norm = []
max_eval = []
for n in num_qubits:
    oH = ObservablesHolmes(n, n, lam_scale)
    Hgad = oH.gadget()
    one_norm.append(np.linalg.norm(qml.matrix(Hgad), ord=1))
    inf_norm.append(np.linalg.norm(qml.matrix(Hgad), ord=np.inf))
    max_eval.append(max(np.abs(qml.eigvals(Hgad))))


fig, ax = plt.subplots()

ax.plot(num_qubits, upper_bounds[2], 'o', label="upper bound")
ax.plot(num_qubits, inf_norm, 's', label=r"$\|H^{gad}\|_{\infty}$")  
ax.plot(num_qubits, one_norm, 'v', label=r"$\|H^{gad}\|_{1}$") 
ax.plot(num_qubits, max_eval, '+', label=r"$\max|\lambda_j|$")  

ax.set_xlabel(r"N Computational Qubits")
ax.set_ylabel(r"$ Norm $")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(r"2-local decomposition of $H^{comp} = \bigotimes_{j=1}^n Z_j$")
ax.legend()

plt.show()