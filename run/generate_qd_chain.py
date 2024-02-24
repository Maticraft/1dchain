import numpy as np

import src.hamiltonian.quantum_dots_chain as qd_chain
from src.hamiltonian.utils import plot_eigvals
from src.plots import plot_matrix

defaults = qd_chain.DefaultParameters()
parameters = qd_chain.QuantumDotsHamiltonianParameters(no_dots=7, no_levels=2, default_parameters=defaults)
system = qd_chain.QuantumDotsHamiltonian(parameters)

print(system.get_label())
h = system.get_hamiltonian()

plot_eigvals(system, 'mu', np.linspace(-1, 1, 100), 'eigvals.png') #, ylim=(-1e-2, 1e-2))
# plot.plot_hamiltonian(hamiltonian)

# eigs = eigh(hamiltonian, eigvals_only=True)

# paramsweep, eigenvalues, occupations = system.parameter_sweeping(parameter_name='mu', start=-1., stop=1., num=101)
# plot.plot_eigenvalues(paramsweep, eigenvalues, occupations, range=[-1.,1.])