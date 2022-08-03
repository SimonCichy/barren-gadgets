import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator

colours = [['#36226b', '#22236b', '#22656b', '#226b2b', '#606b22', '#6b4d22', '#6b2a22'],
           ['#6f45d8', '#4547d8', '#45ccd8', '#45d859', '#c2d845', '#d89d45', '#d85645'],
           ['#977dd8', '#7d7fd8', '#7dd1d8', '#7dd889', '#cbd87d', '#d8b47d', '#d8887d']]

num_qubits = np.arange(2, 7)
k = num_qubits                          # locality
r = 1                                   # number of terms
coefficients = np.ones(r)
lamb = np.arange(0, 1, .1)              # variational parameter λ
norm_Hcomp = np.sum(coefficients)

fig, ax = plt.subplots()

for l in lamb:
    norm_Hgad = r * k * (k - 1) / 2 + l * (np.sum(coefficients) + r * (k - 1))
    ax.plot(num_qubits, norm_Hgad/norm_Hcomp, '.', label=r"$\lambda = {}$".format(l))    

ax.set_xlabel(r"N Computational Qubits")
ax.set_ylabel(r"$H^{gad}/H^{comp}$")
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(r"$\frac{\frac{rk(k-1)}{2} + \lambda \sum_s(c_s+k-1)}{\sum_s c_s}$ for $r=1$, $k=n$")
ax.legend()

plt.show()