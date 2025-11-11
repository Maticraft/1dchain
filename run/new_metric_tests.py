import os
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import torch

from src.data.datasets import HamiltonianFromParametersDataset
from src.hamiltonian.hamiltonian_torch_handlers import HamiltonianConverter, HamiltonianParams
from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.conductance import Transport, generate_conductance_tensor, plot_conductance_map
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonianParameters, QuantumDotsHamiltonian
from src.hamiltonian.utils import plot_eigvals

# mu is negative
# l is not t_phas!!!


test_dir = 'tests_new_metric_3QD'
os.makedirs(test_dir, exist_ok=True)

data_path = './data/quantum_dots/simple_fixed_t_3dots1level_majoranization0_0_mzm_gap0_25'
data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True)


hamiltonian_converter = HamiltonianConverter(
    hamiltonian_params=HamiltonianParams(
        inter_site_real_params=("t", "t_phase_diag", "t_phase_antidiag"),
        inter_site_imag_params=(()),
    ),
    min_inter_site_interaction_range=1,
    max_inter_site_interaction_range=2,
)

h_map = hamiltonian_converter.from_matrix_to_params(data[0][0][0].unsqueeze(0))

# Rashba 3-5QDs
no_dots = 3
defaults = DefaultParameters()
defaults.b_default = 1.837e-05
defaults.d_default = 9.187326213577762e-06
defaults.l_default = 0.0
defaults.l_ksi_default = 0.0
parameters = QuantumDotsHamiltonianParameters(no_dots=no_dots, no_levels=1, default_parameters=defaults)
# parameters.mu = np.array([2.9395860110525973e-05, -1.410196000506403e-05, -2.790976213873364e-05])
# parameters.t = np.array([9.187326213577762e-06, 1.9939523554057814e-05, 0.0])
# parameters.l = np.array([-1.110711693763733, 1.097441554069519, 0.0])

parameters.mu = -np.array([2.9395860110525973e-05, -1.410196000506403e-05, -2.790976213873364e-05])
parameters.t = np.array([9.187326213577762e-06, 1.9939523554057814e-05, 0.0])
parameters.l = np.array([-1.110711693763733, 1.097441554069519, 0.0])


system = QuantumDotsHamiltonian(parameters)

hamiltonian_tensor = system.get_hamiltonian_tensor()
# hamiltonian_tensor = torch.flip(hamiltonian_tensor, dims=[-2, -1])  # flip the last two dimensions to match the expected shape
hamiltonian_tensor = torch.complex(hamiltonian_tensor[0], hamiltonian_tensor[1])


conductance_config = {
    'cmap_list': [
    {
        'cmap2': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'b_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'ef_range': (-1.5/AtomicUnits.Eh, 1.5/AtomicUnits.Eh),
            'b_num': 100,
            'ef_num': 100,
            'with_embedding': False
        },
    },
    {
        'cmap0': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'mu_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
            'ef_range': (-1.5/AtomicUnits.Eh, 1.5/AtomicUnits.Eh),
            'mu_num': 100,
            'ef_num': 100,
            'with_embedding': False,
            'site_index': 0
        }
    },
    {
        'cmap0': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'mu_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
            'ef_range': (-1.5/AtomicUnits.Eh, 1.5/AtomicUnits.Eh),
            'mu_num': 100,
            'ef_num': 100,
            'with_embedding': False,
            'site_index': 1
        }
    },
        {
        'cmap0': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'mu_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
            'ef_range': (-1.5/AtomicUnits.Eh, 1.5/AtomicUnits.Eh),
            'mu_num': 100,
            'ef_num': 100,
            'with_embedding': False,
            'site_index': 2
        }
    }
    ],
}

conductance = generate_conductance_tensor(hamiltonian_tensor, conductance_config)

for i, cmap_config in enumerate(conductance_config['cmap_list']):
    conductance_path = os.path.join(test_dir, 'conductance')
    os.makedirs(conductance_path, exist_ok=True)
    
    cmap_name = list(cmap_config.keys())[0]
    x_name = 'b' if cmap_name == 'cmap2' else 'mu'

    x_tick_range = cmap_config[cmap_name][f'{x_name}_range']
    y_tick_range = cmap_config[cmap_name]['ef_range']
    i_val = cmap_config[cmap_name]['i']
    j_val = cmap_config[cmap_name]['j']
    
    if cmap_name == 'cmap2':
        extra_label = ""
    if cmap_name == 'cmap0':
        extra_label = f'_dot{cmap_config[cmap_name]["site_index"]}'

    plot_conductance_map(
        conductance[i].detach().cpu().numpy(),
        os.path.join(conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{extra_label}.png'),
        xtick_range=x_tick_range,
        ytick_range=y_tick_range,
        xlabel=f"${x_name}$ [mV]",
        ylabel="$E_F$ [meV]"
    )


# hamiltonian = system.full_hamiltonian()

# m3l = np.array([1,1,1,1]+[0,0,0,0]*(no_dots-1)).T/2.
# m3l_ = np.array([1,1,-1,-1]+[0,0,0,0]*(no_dots-1)).T*1.j/2.
# m3r = np.array([0,0,0,0]*(no_dots-1)+[1,1,-1,-1]).T*1.j/2.
# m3r_ = np.array([0,0,0,0]*(no_dots-1)+[1,1,1,1]).T/2.

# m1 = np.concatenate([np.array([1,1,-1,-1]), np.array([0,0,0,0]*(no_dots-2)), np.array([1,1,1,1])]).T/2.
# m2 = np.concatenate([np.array([1,1,-1,-1])*-1., np.array([0,0,0,0]*(no_dots-2)), np.array([1,1,1,1])]).T/2.

# points = []
# majoranas = []
# for v in np.linspace(-1.5/AtomicUnits.Eh, 1.5/AtomicUnits.Eh, num=101):
    
#     system.set_parameter("mu", np.ones(system.parameters.no_dots)*v)
#     hamiltonian = system.full_hamiltonian()
#     eigs, eigvs = eigh(hamiltonian, eigvals_only=False)
#     ms = []
#     for ie, eig in enumerate(eigs):
#         ml = np.conj(eigvs[:,ie])@m1
#         mr = np.conj(eigvs[:,ie])@m2
#         zm = np.exp(-np.abs(eig)/(0.1/AtomicUnits.Eh))
#         points.append([v, eig+ie*0.005/AtomicUnits.Eh, (np.abs(ml)-np.abs(mr))*zm])
#         #points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=0).reshape(2,2).sum(axis=1)[0]])
#         #points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=1)[0]*2])
#         ms.append(np.abs(ml-mr)*zm)  # mode projection on left - right majoranas
#     ms = np.array(ms)[np.abs(eigs).argsort()]  # sort MZM_i using |E_i|
#     majoranas.append([v, np.amax([0., ms[0]+ms[1]-ms[2:].sum()])])  # max(0, MZM_1+MZM_2 - all_others), everything is weighed by exp(-E_i/E_0)

# points = np.array(points)
# majoranas = np.array(majoranas)

# fig, ax = plt.subplots()
# ax.set_xlabel("v_gate")
# ax.set_ylabel("energy")
# ax.set_ylim([-1.,1.])
# cc = ax.scatter(x=points[:,0]*AtomicUnits.Eh, y=points[:,1]*AtomicUnits.Eh, c=points[:,2], s=1., cmap='coolwarm')
# ax.plot(majoranas[:,0]*AtomicUnits.Eh, majoranas[:,1]/2., c='orange')
# cbar = fig.colorbar(cc, ax=ax)
# cbar.set_label(r'majoranization')
# plt.savefig(f'{test_dir}/rashba_3QD.png', dpi=200)
# plt.close()


# system.set_parameter('mu', np.zeros(system.parameters.no_dots)*(1.)/AtomicUnits.Eh)
plot_eigvals(system, 'mu', np.linspace(-.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100), os.path.join(test_dir, 'eigvals_with_occ.png'), color='occupations', xnorm=1./AtomicUnits.Eh, ylim=(-2., 2.), ynorm=1./AtomicUnits.Eh, majoranization=True)
plot_eigvals(system, 'mu', np.linspace(-.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100), os.path.join(test_dir, 'eigvals_with_eh.png'), color='electron-hole-diff', xnorm=1./AtomicUnits.Eh, ylim=(-2., 2.), ynorm=1./AtomicUnits.Eh)
