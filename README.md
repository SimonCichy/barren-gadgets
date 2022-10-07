# Barren gadgets

Implementation of perturbative gadgets for the mitigation of Barren plateaus in 
variational quantum algorithms as presented in 
<a href="https://arxiv.org/abs/2210.03099">this paper</a>.  

## Main idea
Using a perturbation theory to create a perturbative gadget (inspired from 
adiabatic computing). The spectrum of a global Hamitlonian of interest is
encoded within the low-energy subspace of some specially constructed 
Hamiltonian on a larger Hilbert space: the gadget Hamiltonian.
For more details, see the full paper by Cichy, Fährmann et al..

# How to use:

## Perturbative gadgets for own applications
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

/!\ WARNING: The automatic gadgetization using our scripts will work for most simple cases and for the numerical examples presented from the paper, but does
not cover all the extensions and edge cases presented there. 
Please check the gadgetized Hamiltonian (e.g. like done in 
```/barren-gadgets/tests/new_gadget_decomposition_tests.py``` ) 
before running simulations with it.  

## Tutorial on how to use our gadgets
For those interested in the topic for whom it is new, we prepared a tutorial 
based on the <a href="https://pennylane.ai/">Pennylane library</a>. 
It can be found under
```/barren-gadgets/Pennylane-tutorial/perturbative_gadgets_for_VQE.ipynb```.
Just open the notebook and follow the tutorial!

## Reproduction of the plots from the paper
This repository also contains the scripts used to generate the figures from the
paper. For each, the simulation (hence generation of the data) and plotting are
done independently. 
First of all, set where to save the data. To do so, in 
```/barren-gadgets/src/data_management.py```
on lines 7, 38 and 95 set the relative path to where the files should be saved.
```python
data_folder = '../path/to/your/storing/location'
```  
To generate the data from **figure 1**, one needs to run
```/barren-gadgets/own-experiments/gadget_gradients_faehrmann.py```
three times, with the right settings on lines 14-22 and. First 
```python
# General parameters:
generating_Hamiltonian = "global"
num_samples = 1000
layers_list = [2, 5, 10, 20]
qubits_list = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
lambda_scaling = 1                        # w.r.t. lambda_max
gate_set = [qml.RX, qml.RY, qml.RZ]
newk = 3
seed = 43
```
for the target global Hamiltonian,
then
```python
# General parameters:
generating_Hamiltonian = "gadget"
num_samples = 1000
layers_list = [2, 5, 10, 20]
qubits_list = [4, 5, 6, 7, 8, 9]
lambda_scaling = 1                        # w.r.t. lambda_max
gate_set = [qml.RX, qml.RY, qml.RZ]
newk = 3
seed = 43
```
for the 3-local gadget construction and finally
```python
# General parameters:
generating_Hamiltonian = "gadget"
num_samples = 1000
layers_list = [2, 5, 10, 20]
qubits_list = [4, 6, 8, 10, 12]
lambda_scaling = 1                        # w.r.t. lambda_max
gate_set = [qml.RX, qml.RY, qml.RZ]
newk = 4
seed = 43
```
gives the data for the 4-local gadget curves. 
Then, running 
```/barren-gadgets/plotting/paper_plots.py```
having changed lines 11 and 186-188 to reflect the respective location of the
data files and uncommenting only
```
variances_plots()
``` 
in the main loop will save a pdf of the figure under 
```data_folder + '../plots/variances_new_gadget/variances_for_paper.pdf'```.  
*Note*: The data points will not be exactly the same since the data generation
has been splitted in several runs due to access to computational resources, 
hence some data points in the paper resulted from 200 samples batches with 
different random seeds.  

For **figure 2**, the file to generate the data is 
```/barren-gadgets/own-experiments/gadget_scheduled_training.py```
with the settings on lines 12 to 20
```python
use_exact_ground_energy = False
plot_data = False
save_data = True

computational_qubits = 5
newk = 3
max_iter = 500
step = 0.3
num_shots = None
```
and chosing 
```python
schedule = soi.linear_ala_new_gad(pf, opt, max_iter, newk, False)
```
on line 37. Then, to generate the figure itself one can run 
```/barren-gadgets/plotting/paper_plots.py```
updating line 11 and lines 75 to 109 to reflect the location of the files 
generated in the previous step and uncommenting only 
```
training_plots_with_statistics()
``` 
in the main loop. 
You will find the file under
```data_folder + '../plots/training_new_gadget/trainings_for_paper_with_stats.pdf'```

# Repository structure:
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