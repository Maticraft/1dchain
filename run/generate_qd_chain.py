import numpy as np

import src.hamiltonian.quantum_dots_chain as qd_chain
from src.hamiltonian.utils import plot_eigvals, plot_majorana_polarization
from src.plots import plot_matrix

defaults = qd_chain.DefaultParameters()
# defaults.l_ksi_default = 0.3
# defaults.l_rho_default = 0.6
# defaults.l_default = 0.9
parameters = qd_chain.QuantumDotsHamiltonianParameters(no_dots=7, no_levels=2, default_parameters=defaults)
system = qd_chain.QuantumDotsHamiltonian(parameters)

h = system.get_hamiltonian()
vscale = 1/qd_chain.AtomicUnits.Eh

plot_eigvals(system, 'mu', np.linspace(-1.5/qd_chain.AtomicUnits.Eh, .5/qd_chain.AtomicUnits.Eh, 100), 'eigvals.png', ylim=(-1, 1), xnorm=1/qd_chain.AtomicUnits.Eh, ynorm=1/qd_chain.AtomicUnits.Eh)
plot_matrix(h.real, 'hamiltonian_real_1level.png', vmin=-vscale, vmax=vscale)
plot_matrix(h.imag, 'hamiltonian_imag_1level.png', vmin=-0.1*vscale, vmax=0.1*vscale)

system.set_parameter('mu', -0.5/qd_chain.AtomicUnits.Eh)
h = system.get_hamiltonian()
plot_matrix(h.real, 'hamiltonian_wmajoranas_real_1level.png', vmin=-vscale, vmax=vscale)
plot_matrix(h.imag, 'hamiltonian_wmajoranas_imag_1level.png', vmin=-0.1*vscale, vmax=0.1*vscale)
print(system.get_label())
plot_majorana_polarization(system, '.', qd_chain.MZM_THRESHOLD, polaxis='x', string_num=1)

# eigs = eigh(hamiltonian, eigvals_only=True)

# paramsweep, eigenvalues, occupations = system.parameter_sweeping(parameter_name='mu', start=-1., stop=1., num=101)
# plot.plot_eigenvalues(paramsweep, eigenvalues, occupations, range=[-1.,1.])