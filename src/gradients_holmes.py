import pennylane as qml
from pennylane import numpy as np
import datetime

from observables_holmes import ObservablesHolmes
from hardware_efficient_ansatz import HardwareEfficientAnsatz


class GradientsHolmes:
    """ Class containing methods to generate the data to produce the variance
    plots used in the identification of Barren plateaus (variance of the 
    gradients of the cost function as a function of depth in number of qubits)
    
    Args: 
        qubits_list (list)  : list of number of qubits to probe the system on
                              to get one point in the plot
        layer_list (list or string) : one of the following
            · list of the number of layers to probe for each number of qubits
            · string 'linear' to inform to use a circuit depth linear in the 
              number of qubits
        perturbation_factor (float) : parameter controlling the magnitude of the
                                      perturbation in the derivation from 
                                      Jordan2012 (as prefactor to λmax)
        num_samples (int)           : number of samples to take to generate the
                                      statistics of the variance of the gradients
        gate_set (list)             : set of gates to be used in the ansatz
    """
    def __init__(self, qubits_list, layer_list = ['linear'], 
                 perturbation_factor=1, num_samples=200, 
                 gate_set=[qml.RX, qml.RY, qml.RZ]):
        self.qubit_list = qubits_list
        self.layer_list = layer_list
        self.gate_set = gate_set
        self.perturbation_factor = perturbation_factor
        self.num_samples = num_samples
        self.data_folder = '../results/data/gradients/'
    
    def generate(self, circuit):
        """Wrapper function to generate the gradient data in function of the 
        kind of layering. For more details, see called functions: 
        · generate_linear_depth_gradients() 
        · generate_fixed_depth_gradients
        Args: 
            circuit (string): identifier of the circuit to be probed. should be
                              one of the following: 'global', 'local', 'gadget2',
                              'gadget3'
        """
        if self.layer_list == 'linear':
            self.generate_linear_depth_gradients(circuit)
        elif type(self.layer_list[0]) is int:
            self.generate_fixed_depth_gradients(circuit)

    def generate_fixed_depth_gradients(self, circuit):
        """Method generating gradient data for the requested circuit for all 
        combinations of widths and depths as given by self.qubits and 
        self.layers
        """
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
        """Method generating gradient data for the requested circuit for a 
        fixed depth for each width in self.qubits. It uses as many layers as 
        total number of qubits (computational + ancillary)
        """
        file_name = self.get_file_name(circuit)
        with open(file_name, 'w') as of:
            of.write('# layers\t# qubits\tgradients')

        for num_qubits in self.qubit_list:
            print(num_qubits, " qubits")

            if 'gadget' in circuit:
                locality = int(circuit[circuit.find('gadget') + 6])
                num_layers = int(num_qubits * (1 + 1/(locality-1)))
            else:
                num_layers = num_qubits

            print(num_layers, " layers")
            with open(file_name, 'a') as of:
                of.write('\n{}\t{}'.format(num_layers, num_qubits))

            for _ in range(self.num_samples):
                gradient = self.get_gradient(circuit, num_qubits, num_layers)
                with open(file_name, 'a') as of:
                    of.write('\t{}'.format(gradient[0][0]))
        with open(file_name, 'a') as of:
            of.write('\n')
    
    def get_gradient(self, circuit, num_qubits, num_layers,  perturbation_factor=None):
        """Method to generate the cost function of which the gradients will be
        recorded.
        TODO: transform to 'get_gradient(self, circuit, num_qubits, num_layers) 
        -> float ' instead
        
        Args: 
            circuit (string): identifier of the circuit to be probed. should be
                              one of the following: 'global', 'local', 'gadget2',
                              'gadget3'
            num_qubits (int): number of computational qubits (qubits acted upon  
                              by the global Hamiltonian)
            num_layers (int): depth of the circuit to probe
        Returns: 
            cost (callable) : callable function to obtain the differentiable
                              cost value for given parameters (using 
                              qml.ExpvalCost)
            params (list)   : list of (randomly generated) parameters to probe
                              the constructed cost function
        """
        if perturbation_factor == None:
            perturbation_factor = self.perturbation_factor
        if 'gadget' in circuit:
            # finding the target locality of the gadget decomposition
            locality = int(circuit[circuit.find('gadget') + 6])
            assert num_qubits % (locality - 1) == 0, "Non integer number of requested ancillary qubits. The number of computational qubits is not divisible by (locality-1)"
            ancillary_qubits = int(num_qubits / (locality - 1))
            width = num_qubits+ancillary_qubits
            oH = ObservablesHolmes(num_qubits, ancillary_qubits, perturbation_factor)
            obs = oH.gadget()
        else:
            width = num_qubits
            oH = ObservablesHolmes(num_qubits, 0, perturbation_factor)
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
        gradient = qml.grad(cost)(params)
        return gradient
    
    def get_file_name(self, circuit):
        return self.data_folder + '{}_'.format(datetime.datetime.now().strftime("%y%m%d")) + \
              circuit + '_{}qubits_{}layers_{}lambda_{}samples' + \
              '.dat'.format(self.qubit_list[-1], self.layer_list[-1], 
                                                                           self.perturbation_factor, self.num_samples)
        
    def generate_gradients_vs_layers(self, lambda_list, num_qubits, circuit):
        file_name = self.get_file_name(circuit)
        with open(file_name, 'w') as of:
            of.write('# layers\tlambda\tgradients for {} qubits'.format(num_qubits))
        
        for num_layers in self.layer_list:
            print(num_layers, " layers")

            for lam in lambda_list:
                with open(file_name, 'a') as of:
                    of.write('\n{}\t{}'.format(num_layers, lam))
                for _ in range(self.num_samples):
                    gradient = self.get_gradient(circuit, num_qubits, num_layers, lam)
                    with open(file_name, 'a') as of:
                        of.write('\t{}'.format(gradient[0][0]))
        with open(file_name, 'a') as of:
            of.write('\n')


    
