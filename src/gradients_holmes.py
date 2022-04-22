import pennylane as qml
from pennylane import numpy as np
import datetime

from observables_holmes import ObservablesHolmes
from hardware_efficient_ansatz import HardwareEfficientAnsatz


class GradientsHolmes:
    def __init__(self, qubits_list, layer_list = ['linear'], perturbation_factor=1, num_samples=200, gate_set=[qml.RX, qml.RY, qml.RZ]):
        self.qubit_list = qubits_list
        self.layer_list = layer_list
        self.gate_set = gate_set
        self.perturbation_factor = perturbation_factor
        self.num_samples = num_samples
        self.data_folder = '../results/data/gradients/'
    
    def generate(self, circuit):
        if self.layer_list == 'linear':
            self.generate_linear_depth_gradients(circuit)
        elif type(self.layer_list[0]) is int:
            self.generate_fixed_depth_gradients(circuit)

    def generate_fixed_depth_gradients(self, circuit):
        file_name = self.get_file_name(circuit)
        with open(file_name, 'w') as of:
            of.write('# layers\t# qubits\tgradients')

        for num_qubits in self.qubit_list:
            print(num_qubits, " qubits")

            for num_layers in self.layer_list:
                print(num_layers, " layers")
                with open(file_name, 'a') as of:
                    of.write('\n{}\t{}'.format(num_layers, num_qubits))

                for _ in range(self.num_samples):
                    cost, params = self.get_cost(circuit, num_qubits, num_layers)
                    gradient = qml.grad(cost)(params)
                    with open(file_name, 'a') as of:
                        of.write('\t{}'.format(gradient[0][0]))
        with open(file_name, 'a') as of:
            of.write('\n')
    
    def generate_linear_depth_gradients(self, circuit):
        file_name = self.get_file_name(circuit)
        with open(file_name, 'w') as of:
            of.write('# layers\t# qubits\tgradients')

        for num_qubits in self.qubit_list:
            print(num_qubits, " qubits")

            locality = int(circuit[circuit.find('gadget') + 6])
            num_layers = int(num_qubits * (1 + 1/(locality-1)))

            print(num_layers, " layers")
            with open(file_name, 'a') as of:
                of.write('\n{}\t{}'.format(num_layers, num_qubits))

            for _ in range(self.num_samples):
                cost, params = self.get_cost(circuit, num_qubits, num_layers)
                gradient = qml.grad(cost)(params)
                with open(file_name, 'a') as of:
                    of.write('\t{}'.format(gradient[0][0]))
        with open(file_name, 'a') as of:
            of.write('\n')
    
    def get_cost(self, circuit, num_qubits, num_layers):
        if 'gadget' in circuit:
            locality = int(circuit[circuit.find('gadget') + 6])
            assert num_qubits % (locality - 1) == 0, "Non integer numer of requested ancillary qubits. The number of computational qubits is not divisible by (locality-1)"
            ancillary_qubits = int(num_qubits / (locality - 1))
            width = num_qubits+ancillary_qubits
            oH = ObservablesHolmes(num_qubits, ancillary_qubits, self.perturbation_factor)
            obs = oH.gadget()
        else:
            width = num_qubits
            oH = ObservablesHolmes(num_qubits, 0, self.perturbation_factor)
            if circuit == "global":
                obs = oH.computational()
            elif circuit == "local":
                obs = oH.local()
        params = np.random.uniform(0, np.pi, size=(num_layers, width))
        random_gate_sequence = [[np.random.choice(self.gate_set) for _ in range(width)] for _ in range(num_layers)]
        dev = qml.device("default.qubit", wires=range(width))     # /!\ only for r=1, k=n
        hea = HardwareEfficientAnsatz(random_gate_sequence)
        ansatz = hea.ansatz
        cost = qml.ExpvalCost(ansatz, obs, dev)
        return cost, params
    
    def get_file_name(self, circuit):
        return self.data_folder + '{}_'.format(datetime.datetime.now().strftime("%y%m%d")) + \
              circuit + '_{}qubits_{}layers_{}lambda_{}samples.dat'.format(self.qubit_list[-1], self.layer_list[-1], 
                                                                           self.perturbation_factor, self.num_samples)


    
