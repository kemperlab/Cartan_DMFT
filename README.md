[![DOI](https://zenodo.org/badge/452008057.svg)](https://zenodo.org/badge/latestdoi/452008057)

# Mapping the metal-insulator phase diagram by algebraically fast-forwarding dynamics on a cloud quantum computer

Thomas Steckmann, Trevor Keen, Efekan Kokcu,    Alexander F. Kemper, Eugene F. Dumitrescu, Yan Wang

[arXiv:2112.05688](https://arxiv.org/abs/2112.05688)

*Abstract:*

Dynamical mean-field theory (DMFT) maps the local Greens function of the Hubbard model to that of the Anderson impurity model and thus gives an approximate solution of the Hubbard model by solving the simpler quantum impurity model. Quantum and hybrid quantum-classical algorithms have been proposed to efficiently solve impurity models by preparing and evolving the ground state under the impurity Hamiltonian on a quantum computer instead of using intractable classical algorithms. We propose a highly optimized fast-forwarding quantum circuit to significantly improve quantum algorithms for the minimal DMFT problem preserving the Mott phase transition.Our Cartan decomposition based algorithm uses a fixed depth quantum circuit to eliminate time-discretization errors and evolve the initial state over arbitrary times. By exploiting the structure of the fast-forwarding circuits we reduce the gate count (77 CNOTs after optimization), simulate the dynamics, and extract frequencies from the Anderson impurity model on noisy quantum hardware.We then demonstrate the Mott transition by mapping the phase-diagram of the corresponding impurity problem.  Especially near the Mott phase transition when the quasiparticle resonance frequency converges to zero and evolving the system over long-time scales is necessary, our method maintains accuracy where Trotter error would otherwise dominate.  This work presents the first computation of the Mott phase transition using noisy digital quantum hardware, made viable by a highly optimized computation in terms of gate depth, simulation error, and run-time on quantum hardware. The combination of algebraic circuit decompositions and model specific error mitigation techniques used may have applications extending beyond our use case to solving correlated electronic phenomena on noisy quantum computers.

---

This repositiory contains:

- The data supporting the results of the paper
- The source code used to generate and execute circuits on IBM quantum hardware using a free account
- Data processing files to compute the phase diagrams and supporting figures

**Requirements**

OpenFermion

Qiskit <= 0.32.1

[Cartan Quantum Synthesizer](https://github.com/kemperlab/cartan-quantum-synthesizer)

