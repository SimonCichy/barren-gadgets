# Barren gadgets

Implementation of perturbative gadgets for the mitigation of Barren plateaus in 
variational quantum algorithms  

## Main idea
Using a perturbation theory to create a perturbative gadget (inspired from 
adiabatic computing). The spectrum of a global Hamitlonian of interest is
encoded within the low-energy subspace of some specially constructed 
Hamiltonian on a larger Hilbert space: the gadget Hamiltonian.
For more details, see the full paper by Cichy, Fährmann et al..

## Repository structure:
- analytic: some Mathematica scripts that were used to understand the theory of 
perturbative gadgets
- obsolete: old versions of scripts that are not up to date anymore (and neither
maintained), kept for the sake of verification if needed. To be eliminated. 
- *own-experiments:* compilation of scripts and notebooks corresponding to 
experiments on perturbative gadgets that have been re-written using more 
abstraction
  - gadget_gradients_faehrmann.py: main script to generate the data on gradient 
  variances. 
  - gadget_scheduled_training.py: script to run various kind of training 
  simulations. Strongly relies on src/training/* for the running of the 
  simulation (scheduled_training()) and for the definitions of the 
  simulation settings (SchedulesOfInterest())  
- *src:* 
  folder with the source files containing methods used in the main scripts  
  - data_management: 
    methods used for saving, reading and recovering the data
    from the gradients calculations and training simulations
  - faehrmann_gadgets: 
    automated generation of the gadget Hamiltonian according to Cichy, Fährmann 
    et al. for any given computational Hamiltonian as <a href="https://pennylane.
    readthedocs.io/en/stable/code/api/pennylane.Hamiltonian.html">qml.
    Hamiltonian</a>
  - hardware_efficient_ansatz: 
    compilation of classes falling under the "hardware efficient ansatz" 
    description. Each has a method self.ansatz() that can be used with <a href
    ="https://pennylane.readthedocs.io/en/stable/code/api/pennylane.ExpvalCost.
    html">qml.ExpvalCost</a> 
    - AlternatingLayeredAnsatz():
      as used by <a href="https://www.nature.com/articles/s41467-018-07090-4">
      McClean et al.</a> and <a href="http://arxiv.org/abs/2101.02138">Holmes et
       al.</a>. 
    - SimplifiedAlternatingLayeredAnsatz(): 
      as used in <a href="https://www.nature.com/articles/s41467-021-21728-w">
      Cerezo et al.</a>, similar to <a href="https://pennylane.readthedocs.io/
      en/latest/code/api/pennylane.SimplifiedTwoDesign.html#pennylane.Simplified
      TwoDesign">qml.SimplifiedTwoDesign</a>    
    - HardwareEfficientAnsatz(): **deprecated**
  - jordan_gadgets: 
    automated generation of the gadget Hamiltonian according to <a href="https:
    //link.aps.org/doi/10.1103/PhysRevA.77.062329">Jordan & Fahri</a> for any 
    given computational Hamiltonian as <a href="https://pennylane.
    readthedocs.io/en/stable/code/api/pennylane.Hamiltonian.html">qml.
    Hamiltonian</a>
  - merge_files: 
    script used to merge several data files into a single one when the 
    simulations where splitted into several runs
  - observables_holmes: (**deprecated**)
    class to generate the relevant observables related to the application of
    the perturbative gadgets from <a href="https://link.aps.org/doi/10.1103/
    PhysRevA.77.062329">Jordan & Fahri</a> to the examples from <a href="http:/
    /arxiv.org/abs/2101.02138">Holmes et al.</a>. 
  - trainings: 
    script containing the main method to run training simulations with 
    schedules and a collection of relevant schedules for our work 
- *plotting:* 
  directory containing the notebooks displaying the main results
- *tests*: 
  some test scripts to make sure that the main source files keep being 
  correct implementations when updating them

## How to use:
The method of main interest to the user will probably be that of automatic 
generation of the gadget Hamiltonian from a given target Hamiltonian. 
The method is gadgetize() from the NewPerturbativeGadgets class. 
To use it, first import the relevant packages
```python
import pennylane as qml
from faehrmann_gadgets import NewPerturbativeGadgets
```
then create the (global) hamiltonian of interest. It should be built using the
<a href="https://pennylane.readthedocs.io/en/stable/code/api/pennylane.
Hamiltonian.html">qml.Hamiltonian</a> class and strings of single qubit 
operators. For example, the linear combination of two Pauli words
```python
term1 = qml.operation.Tensor(qml.PauliX(0), qml.PauliX(1), qml.PauliY(2), qml.PauliZ(3))
term2 = qml.operation.Tensor(qml.PauliZ(0), qml.PauliY(1), qml.PauliX(2), qml.PauliX(3))
Hcomp = qml.Hamiltonian([0.4, 0.7], [term1, term2])
```
Next, create the gadgetizer object with the desired settings and generate
the gadget Hamiltonian
```python
gadgetizer = NewPerturbativeGadgets(perturbation_factor=1)
Hgad = gadgetizer.gadgetize(Hamiltonian=Hcomp, target_locality=3)
```