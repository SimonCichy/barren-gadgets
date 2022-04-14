import sys
sys.path.append('../src')
sys.path.append('src')
import pennylane as qml
from pennylane import numpy as np
import datetime

# from gadget_cost import GadgetCost
from observables_holmes import ObservablesHolmes
from hardware_efficient_ansatz import HardwareEfficientAnsatz

np.random.seed(42)
data_folder = '../results/data/gradients/'

data_to_produce = 'variance vs qubits'
# data_to_produce = 'variance vs layers'

# General parameters:
num_samples = 200
layers_list = [1, 2, 5, 10, 20, 50]         # [1, 2, 5, 10, 20, 50]
# If data_to_produce == 'variance vs qubits'
qubits_list = [2, 4, 6, 8]               # [2, 3, 4, 5, 6]
lambda_scaling = 0.5                        # w.r.t. λ_max

# ansatz = qml.templates.StronglyEntanglingLayers

gate_set = [qml.RX, qml.RY, qml.RZ]

def generate_gradients_vs_qubits(layer_list, qubit_list, circuit):
    file_name = data_folder + '{}_'.format(datetime.datetime.now().strftime("%y%m%d")) + \
              circuit + '_{}qubits_{}layers_{}lambda_{}samples.dat'.format(qubit_list[-1], layer_list[-1], lambda_scaling, num_samples)
    with open(file_name, 'w') as of:
        of.write('# layers\t# qubits\tgradients')

    for num_layers in layer_list:
        print(num_layers, " layers")
        #TODO: change to progress bar

        for num_qubits in qubit_list:
            print(num_qubits, " qubits")
            #TODO: change to progress bar

            # write a new line
            with open(file_name, 'a') as of:
                of.write('\n{}\t{}'.format(num_layers, num_qubits))

            for _ in range(num_samples):
                if 'gadget' in circuit:
                    locality = int(circuit[circuit.find('gadget') + 6])
                    ancillary_qubits = num_qubits / (locality - 1)
                    # Generating the random values for the rotations
                    params = np.random.uniform(0, np.pi, size=(num_layers, num_qubits+ancillary_qubits))
                    # params = np.random.uniform(0, np.pi, size=(num_layers, 2*num_qubits, 3))
                    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(num_qubits+ancillary_qubits)] for _ in range(num_layers)]
                    dev = qml.device("default.qubit", wires=range(num_qubits+ancillary_qubits))    # /!\ only for r=1, k=n
                    oH = ObservablesHolmes(num_qubits, ancillary_qubits, lambda_scaling)
                    # cf = GadgetCost(num_qubits, 2*num_qubits, dev)
                    obs = oH.gadget()
                else:
                    params = np.random.uniform(0, np.pi, size=(num_layers, num_qubits))
                    # params = np.random.uniform(0, np.pi, size=(num_layers, num_qubits, 3))
                    random_gate_sequence = [[np.random.choice(gate_set) for _ in range(num_qubits)] for _ in range(num_layers)]
                    dev = qml.device("default.qubit", wires=range(num_qubits))
                    oH = ObservablesHolmes(num_qubits, 0, lambda_scaling)
                    # cf = GadgetCost(num_qubits, num_qubits, dev)
                    if circuit == "global":
                        obs = oH.computational()
                    elif circuit == "local":
                        obs = oH.local()
                
                hea = HardwareEfficientAnsatz(random_gate_sequence)
                ansatz = hea.ansatz

                cost = qml.ExpvalCost(ansatz, obs, dev)
                # Calculating the gradients of the cost function w.r.t the parameters
                # gradient = qml.grad(cf.cost_function)(params, gate_sequence=random_gate_sequence, observable=obs)
                gradient = qml.grad(cost)(params)

                # Write each newly calculated value (innefficient?)
                with open(file_name, 'a') as of:
                    of.write('\t{}'.format(gradient[0][0]))
                    # of.write('\t{}'.format(gradient[0][0][1]))
                # print(gradient[0][0][0])
                    
    # End file with one last line-break
    with open(file_name, 'a') as of:
        of.write('\n')



if data_to_produce == 'variance vs qubits':
    print("Global circuit: ")
    generate_gradients_vs_qubits(layers_list, qubits_list, "global")
    print("Local circuit: ")
    generate_gradients_vs_qubits(layers_list, qubits_list, "local")
    print("2-local gadget circuit: ")
    generate_gradients_vs_qubits(layers_list, qubits_list, "gadget2")





