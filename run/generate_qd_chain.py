import os
import numpy as np

from src.hamiltonian.hamiltonian import RepresentationMapping, REPRESENTATION_MAPPING_FUNCTION
import src.hamiltonian.quantum_dots_chain as qd_chain
from src.hamiltonian.utils import plot_eigvals, plot_majorana_polarization
from src.plots import plot_matrix

dir_name = 'test_representation_old'
os.makedirs(dir_name, exist_ok=True)

defaults = qd_chain.DefaultParameters()
defaults.l_ksi_default = 2.3
defaults.l_rho_default = 1.6
defaults.l_default = 1.9
parameters = qd_chain.QuantumDotsHamiltonianParameters(no_dots=7, no_levels=2, default_parameters=defaults)
system = qd_chain.QuantumDotsHamiltonian(parameters)

h_org = system.get_hamiltonian()
vscale = 1/qd_chain.AtomicUnits.Eh

plot_eigvals(system, 'mu', np.linspace(-1.5/qd_chain.AtomicUnits.Eh, .5/qd_chain.AtomicUnits.Eh, 100), os.path.join(dir_name, 'eigvals.png'), ylim=(-1, 1), xnorm=1/qd_chain.AtomicUnits.Eh, ynorm=1/qd_chain.AtomicUnits.Eh)
plot_eigvals(system, 'mu', np.linspace(-1.5/qd_chain.AtomicUnits.Eh, .5/qd_chain.AtomicUnits.Eh, 100), os.path.join(dir_name, 'eigvals_majorana_rep.png'), ylim=(-1, 1), xnorm=1/qd_chain.AtomicUnits.Eh, ynorm=1/qd_chain.AtomicUnits.Eh, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)

plot_matrix(h_org.real, os.path.join(dir_name, 'hamiltonian_real_2levels.png'), vmin=-vscale, vmax=vscale)
plot_matrix(h_org.imag, os.path.join(dir_name, 'hamiltonian_imag_2levels.png'), vmin=-0.1*vscale, vmax=0.1*vscale)

h = system.get_hamiltonian(representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
plot_matrix(h.real, os.path.join(dir_name, 'hamiltonian_real_2levels_majorana_rep.png'), vmin=-vscale, vmax=vscale)
plot_matrix(h.imag, os.path.join(dir_name, 'hamiltonian_imag_2levels_majorana_rep.png'), vmin=-0.1*vscale, vmax=0.1*vscale)
h_transformed_to_org = REPRESENTATION_MAPPING_FUNCTION[RepresentationMapping.inverse_majorana_plus_minus_up_down](h)
plot_matrix(h_transformed_to_org.real, os.path.join(dir_name, 'hamiltonian_real_2levels_majorana_rep_inverse.png'), vmin=-vscale, vmax=vscale)
plot_matrix(h_transformed_to_org.imag, os.path.join(dir_name, 'hamiltonian_imag_2levels_majorana_rep_inverse.png'), vmin=-0.1*vscale, vmax=0.1*vscale)


system.set_parameter('mu', -0.5/qd_chain.AtomicUnits.Eh)
h = system.get_hamiltonian()
plot_matrix(h.real, os.path.join(dir_name, 'hamiltonian_wmajoranas_real_2levels.png'), vmin=-vscale, vmax=vscale)
plot_matrix(h.imag, os.path.join(dir_name, 'hamiltonian_wmajoranas_imag_2levels.png'), vmin=-0.1*vscale, vmax=0.1*vscale)

h = system.get_hamiltonian(representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
plot_matrix(h.real, os.path.join(dir_name, 'hamiltonian_wmajoranas_real_2levels_majorana_rep.png'), vmin=-vscale, vmax=vscale)
plot_matrix(h.imag, os.path.join(dir_name, 'hamiltonian_wmajoranas_imag_2levels_majorana_rep.png'), vmin=-0.1*vscale, vmax=0.1*vscale)

print(system.get_label())
polarization_dir = os.path.join(dir_name, 'polarization')
os.makedirs(polarization_dir, exist_ok=True)
plot_majorana_polarization(system, polarization_dir, qd_chain.MZM_THRESHOLD, polaxis='y', string_num=1, representation_mapping=RepresentationMapping.second_quantized_plus_minus_up_down)

polarization_dir = os.path.join(dir_name, 'polarization_majorana_rep')
os.makedirs(polarization_dir, exist_ok=True)
plot_majorana_polarization(system, polarization_dir, qd_chain.MZM_THRESHOLD, polaxis='x', string_num=1, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)

# eigs = eigh(hamiltonian, eigvals_only=True)

# paramsweep, eigenvalues, occupations = system.parameter_sweeping(parameter_name='mu', start=-1., stop=1., num=101)
# plot.plot_eigenvalues(paramsweep, eigenvalues, occupations, range=[-1.,1.])