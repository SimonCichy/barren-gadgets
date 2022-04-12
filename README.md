# Barren gadgets

Implementation of perturbative gadgets for the mitigation of Barren plateaus in variational quantum algorithms  

## Main idea
Following the derivation from Jordan et al. (2012), starting from a target k-local Hamiltonian:  
    $$H^{comp} = \sigma_{1} \sigma_{2} \dots \sigma_{n} $$
Using the corresponding gadget Hamiltonian:  
    $$H^{gad} = H^{anc} + V $$
with $H^{anc} = \sum\limits_{1\leq i \leq j \leq k} \frac{1}{2}(\mathbb{I} - Z_{i}Z_{j}) $
and $V = \sum\limits_{j=1}^k c_{j} \sigma_{j}\otimes X_{j}$  
one obtains that the shifted effective Hamiltonian on the low-energy subspace of the gadget Hamiltonian acting on the +1 eigenspace if $X^{\otimes n}$ behaves like the computational Hamiltonian
    $$\tilde{H}_{eff}(H_+^{gad}, 2^n, f(\lambda)) = \frac{-k(-\lambda)^k}{(k-1)!} H^{comp} \otimes P_+ + \mathcal{O}(\lambda^{k+1}) $$  

Looking again at the example Hamiltonian used by Holmes et al.
    $$H_G = \bigotimes_{i=1}^n \sigma_i^z $$
which has $r=1$ and $k=n$. For the example of $n=4$ one obtains: 
    $$H^{gad} = H^{anc} + V 
    = (\mathbb{I} - Z_1^{(a)} Z_2^{(a)}) + (\mathbb{I} - Z_1^{(a)} Z_3^{(a)}) + (\mathbb{I} - Z_1^{(a)} Z_4^{(a)})
    + (\mathbb{I} - Z_2^{(a)} Z_3^{(a)}) + (\mathbb{I} - Z_2^{(a)} Z_4^{(a)}) + (\mathbb{I} - Z_3^{(a)} Z_4^{(a)}) 
    + Z_1^{(c)} \otimes X_1^{(a)} + Z_2^{(c)} \otimes X_2^{(a)} + Z_3^{(c)} \otimes X_3^{(a)} + Z_4^{(c)} \otimes X_4^{(a)}$$

## Repository structure:
- [to be eliminated] pennylane-tutorials: set of demos downloaded from the pennylane website to use as references (unedited)  
- [to be eliminated] adapted tutorials: pennylane demos that have been altered for learning purposes to get used to pennylane  
- own-experiments: compilation of scripts and notebooks corresponding to experiments on perturbative gadgets 
  - [runable] gadget_training_Holmes.py: training experiment using 2-local gadget decomposition to optimize the computational Hamiltonian  
  - [runable] gadget_training_tests.py: simple tests to check the correct implementation of some utilitary scripts  
  - [runable] paper_perturbative_gadgets_generate.ipynb: generation of the gradient variance data  
  - [utils] gadget_gradients_utils.py: collection of methods used in the gradient generation notebook  
  - [utils] gadget_training_utils.py: collection of methods used in the training script (to be replaced by gadget_cost.py + observables_holmes.py)  
  - [utils] gadget_cost: class that implements the calculation of the expectation value of a given observable on the gadget circuit
  - [utils] obsrvables_holmes.py: class that generates the relevant observables for the example from the Holmes2021 paper   
- plotting: directory containing the notebooks displaying the main results and necessary utils  

## TODOs:
- Rewrite the gradient generation to accept an arbitrary observable (WET -> DRY)
