import sys
sys.path.append('../src')
sys.path.append('src')
import numpy as np
import matplotlib.pyplot as plt

from gradients_holmes import GradientsHolmes

np.random.seed(42)

# General parameters:
num_samples = 200
layers_list = [1, 2, 5, 10, 20, 50]         # [1, 2, 5, 10, 20, 50]
# layers_list = 'linear'
qubits_list = [2, 4, 6, 8]               # [2, 3, 4, 5, 6]
lambda_scaling = 0.5                        # w.r.t. λ_max

if __name__ == "__main__":
    gh = GradientsHolmes(qubits_list, layers_list, lambda_scaling, num_samples)
    # print("Global circuit: ")
    # gh.generate("global")
    # print("Local circuit: ")
    # gh.generate("local")
    # print("3-local gadget circuit: ")
    # gh.generate("gadget3")
    # print("2-local gadget circuit: ")
    # gh.generate("gadget2")
    print("gradients vs layers")
    gh.generate_gradients_vs_layers(lambda_list=np.linspace(0, 1, 6), 
                                    num_qubits=4, circuit="gadget2")





