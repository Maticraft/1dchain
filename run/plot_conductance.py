import os

import numpy as np
import matplotlib.pyplot as plt

from src.hamiltonian.conductance import Transport, plot_conductance_map
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonianParameters, QuantumDotsHamiltonian
from src.hamiltonian.quantum_dots_chain import AtomicUnits as au


plot_dir = './test_conductance'
plot_name = 'conductance_map_{}.png'
os.makedirs(plot_dir, exist_ok=True)
plot_path = os.path.join(plot_dir, plot_name)

defaults = DefaultParameters()
defaults.l_ksi_default = 2.3
defaults.l_rho_default = 1.6
defaults.l_default = 1.9

parameters = QuantumDotsHamiltonianParameters(no_dots=3, no_levels=1, default_parameters=defaults)
system = QuantumDotsHamiltonian(parameters)

transport = Transport(system, gamma=.1)

ef_range = (-2./au.Eh, 2./au.Eh)
mu_range = (-2./au.Eh, 2./au.Eh)
C_map = transport.c_map0(0, 0, ef_range, mu_range) # shape (N, 3), with x values in the first column, y values in the second column, and the actual conductance values in the third column
# transform c_map to a 2D array with shape (N, N)

plot_conductance_map(C_map, filename=plot_path.format("mu_Ef"), xtick_range=ef_range, ytick_range=mu_range, xlabel="$V$ [mV]", ylabel="$E_F$ [meV]")

C_map = transport.c_map1(0, 0, mu_range)
plot_conductance_map(C_map, filename=plot_path.format("mul_mur"), xtick_range=mu_range, ytick_range=mu_range, xlabel="$V_L$ [mV]", ylabel="$V_R$ [mV]")