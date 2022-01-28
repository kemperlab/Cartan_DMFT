[![DOI](https://zenodo.org/badge/452008057.svg)](https://zenodo.org/badge/latestdoi/452008057)

# Mapping the metal-insulator phase diagram by algebraically fast-forwarding dynamicson a cloud quantum computer

Thomas Steckmann, Trevor Keen, Alexander F. Kemper, Eugene F. Dumitrescu, Yan Wang

[arXiv:2112.05688](https://arxiv.org/abs/2112.05688)

*Abstract:*

Dynamical mean-field theory (DMFT) maps the local Greens function of the Hubbard modelto that of the Anderson impurity model and thus gives an approximate solution of the Hubbardmodel by solving the simpler quantum impurity model. Quantum and hybrid quantum-classicalalgorithms have been proposed to efficiently solve impurity models by preparing and evolving theground state under the impurity Hamiltonian on a quantum computer instead of using intractableclassical algorithms. We propose a highly optimized fast-forwarding quantum circuit to significantlyimprove quantum algorithms for the minimal DMFT problem preserving the Mott phase transition.Our Cartan decomposition based algorithm uses a fixed depth quantum circuit to eliminate time-discretization errors and evolve the initial state overarbitrarytimes. By exploiting the structure ofthe fast-forwarding circuits we reduce the gate count (77 CNOTs after optimization), simulate thedynamics, and extract frequencies from the Anderson impurity model on noisy quantum hardware.We then demonstrate the Mott transition by mapping the phase-diagram of the correspondingimpurity problem.  Especially near the Mott phase transition when the quasiparticle resonancefrequency converges to zero and evolving the system over long-time scales is necessary, our methodmaintains accuracy where Trotter error would otherwise dominate.  This work presents the firstcomputation of the Mott phase transition using noisy digital quantum hardware, made viable by ahighly optimized computation in terms of gate depth, simulation error, and run-time on quantumhardware. The combination of algebraic circuit decompositions and model specific error mitigationtechniques used may have applications extending beyond our use case to solving correlated electronicphenomena on noisy quantum computers.

---

This repositiory contains:

- The data supporting the results of the paper
- The source code used to generate and execute circuits on IBM quantum hardware using a free account
- Data processing files to compute the phase diagrams and supporting figures

** Requirements**
OpenFermion
Qiskit <= 0.32.1
[Cartan Quantum Synthesizer](https://github.com/kemperlab/cartan-quantum-synthesizer)
