# Barren gadgets

Implementation of perturbative gadgets for the mitigation of Barren plateaus in 
variational quantum algorithms  

## Main idea
Following the derivation from Jordan et al. (2012), starting from a target 
k-local Hamiltonian:  
    $$H^{comp} = \sigma_{1} \sigma_{2} \dots \sigma_{n} $$
Using the corresponding gadget Hamiltonian:  
    $$H^{gad} = H^{anc} + V $$
with $H^{anc} = \sum\limits_{1\leq i \leq j \leq k} \frac{1}{2}(\mathbb{I} - Z_{i}Z_{j}) $
and $V = \sum\limits_{j=1}^k c_{j} \sigma_{j}\otimes X_{j}$  
one obtains that the shifted effective Hamiltonian on the low-energy subspace 
of the gadget Hamiltonian acting on the +1 eigenspace if $X^{\otimes n}$ 
behaves like the computational Hamiltonian
    $$\tilde{H}_{eff}(H_+^{gad}, 2^n, f(\lambda)) = 
    \frac{-k(-\lambda)^k}{(k-1)!} H^{comp} \otimes P_+ 
    + \mathcal{O}(\lambda^{k+1}) $$  

Looking again at the example Hamiltonian used by Holmes et al.
    $$H_G = \bigotimes_{i=1}^n \sigma_i^z $$
which has $r=1$ and $k=n$. For the example of $n=4$ one obtains: 
    $$H^{gad} = H^{anc} + V 
    = (\mathbb{I} - Z_1^{(a)} Z_2^{(a)}) + (\mathbb{I} - Z_1^{(a)} Z_3^{(a)}) 
    + (\mathbb{I} - Z_1^{(a)} Z_4^{(a)})
    + (\mathbb{I} - Z_2^{(a)} Z_3^{(a)}) + (\mathbb{I} - Z_2^{(a)} Z_4^{(a)}) 
    + (\mathbb{I} - Z_3^{(a)} Z_4^{(a)}) 
    + Z_1^{(c)} \otimes X_1^{(a)} + Z_2^{(c)} \otimes X_2^{(a)} 
    + Z_3^{(c)} \otimes X_3^{(a)} + Z_4^{(c)} \otimes X_4^{(a)}$$

## Repository structure:
- ~~[to be eliminated] pennylane-tutorials:~~ set of demos downloaded from the 
pennylane website to use as references (unedited)  
- ~~[to be eliminated] adapted tutorials:~~ pennylane demos that have been 
altered for learning purposes to get used to pennylane  
- [to be eliminated] *own-experiments-hardcoded:* compilation of scripts and 
notebooks corresponding to experiments on perturbative gadgets 
  - [runable] gadget_training_Holmes.py: training experiment using 2-local 
  gadget decomposition to optimize the computational Hamiltonian  
  - [runable] paper_perturbative_gadgets_generate.ipynb: generation of the 
  gradient variance data  
  - [utils] gadget_gradients_utils.py: collection of methods used in the 
  gradient generation notebook  
  - [utils] gadget_training_utils.py: collection of methods used in the 
  training script (to be replaced by gadget_cost.py + observables_holmes.py)   
- *own-experiments:* compilation of scripts and notebooks corresponding to 
experiments on perturbative gadgets that have been re-written using more 
abstraction
  - [runable] gadget_training_Holmes.py: training experiment using 2-local or 
  3-local gadget decomposition to optimize the computational Hamiltonian  
  - [runable] gadget_gradient_holmes.py: generation of the gradient variance 
  data (using the classes instead of the hard-coded functions)
- *src:* folder with the source files containing methods used in the main 
scripts  
  - hardware_efficient_ansatz: compilation of classes falling under the 
  "hardware efficient ansatz" description (for now only one to be renamed 
  AlternatingLayeredAnsatz). Has a method self.ansatz that can be used with 
  qml.ExpvalCost
  - observables_holmes.py: class that generates the relevant observables for 
  the example from the Holmes2021 paper  
  - gradient_holmes.py: class containing methods to generate the data of the 
  variance vs qubits plots
  - observable_norms.py: file containing methods to numerically generate the 
  norms/eigenvalues of the studied operators (to be moved to ownexperiments)
  gadget_plots.py: compilation of methods used to generate the different plots 
  in relevant_results.ipynb
  - gadget_plots.py: some plotting methods specifically created for the 
  different outputs of the data generating scripts
  jordan_gadgets: class to automatically generate an effective Hamiltonian 
  given a target Hamiltonian according to the method from the Jordan paper  
- *plotting:* directory containing the notebooks displaying the main results
- *tests*: some test scripts to make sure that the main source files keep being 
correct implementations when updating them
  - [runable] gadget_ansatz_tests.py 
  - [runable] gadget_decomposition_tests.py: testing the jordan_gadgets.py
  - [runable] gadget_matrix_tests.py 

## TODOs:
- Rewrite the gradient generation to accept an arbitrary observable (WET -> DRY)
- Create a base class for the observables (virtual global and gadget, but runnable projectors)
- Create a more flexible training method that accepts a schedule to be able to change depth, measured observable, ... during training
- Create a method that gets a qml.Hamiltonian object and automatically decomposes it into the gadgetised equivalent
