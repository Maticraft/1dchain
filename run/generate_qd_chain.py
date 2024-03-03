import numpy as np

import src.hamiltonian.quantum_dots_chain as qd_chain
from src.hamiltonian.utils import plot_eigvals
from src.plots import plot_matrix

defaults = qd_chain.DefaultParameters()
parameters = qd_chain.QuantumDotsHamiltonianParameters(no_dots=7, no_levels=2, default_parameters=defaults)
system = qd_chain.QuantumDotsHamiltonian(parameters)

# print(system.get_label())

h = system.get_hamiltonian()
vscale = 1/qd_chain.AtomicUnits.Eh

plot_eigvals(system, 'mu', np.linspace(-1.5/qd_chain.AtomicUnits.Eh, .5/qd_chain.AtomicUnits.Eh, 100), 'eigvals.png', ylim=(-1, 1), xnorm=1/qd_chain.AtomicUnits.Eh, ynorm=1/qd_chain.AtomicUnits.Eh)
plot_matrix(h.real, 'hamiltonian_real.png', vmin=-vscale, vmax=vscale)
plot_matrix(h.imag, 'hamiltonian_imag.png', vmin=-0.1*vscale, vmax=0.1*vscale)

# eigs = eigh(hamiltonian, eigvals_only=True)

# paramsweep, eigenvalues, occupations = system.parameter_sweeping(parameter_name='mu', start=-1., stop=1., num=101)
# plot.plot_eigenvalues(paramsweep, eigenvalues, occupations, range=[-1.,1.])