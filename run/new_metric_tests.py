import os
import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import torch

from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.conductance import generate_conductance_tensor, plot_conductance_map
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonianParameters, QuantumDotsHamiltonian


test_dir = 'tests_new_metric_3QD'
os.makedirs(test_dir, exist_ok=True)

# Rashba 3-5QDs
no_dots = 3
defaults = DefaultParameters()
parameters = QuantumDotsHamiltonianParameters(no_dots=no_dots, no_levels=1, default_parameters=defaults)
system = QuantumDotsHamiltonian(parameters)

system.set_parameter('mu', np.ones(system.parameters.no_dots)*(1.)/AtomicUnits.Eh)
hamiltonian_tensor = system.get_hamiltonian_tensor()
hamiltonian_tensor = torch.complex(hamiltonian_tensor[0], hamiltonian_tensor[1])


conductance_config = {
    'cmap_list': [
    {
        'cmap2': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'b_range': (0./AtomicUnits.Eh, 1./AtomicUnits.Eh),
            'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
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
            'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
            'mu_num': 100,
            'ef_num': 100,
            'with_embedding': False
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
    plot_conductance_map(
        conductance[i].detach().cpu().numpy(),
        os.path.join(conductance_path, f'{cmap_name}_i{i_val}_j{j_val}.png'),
        xtick_range=x_tick_range,
        ytick_range=y_tick_range,
        xlabel=f"${x_name}$ [mV]",
        ylabel="$E_F$ [meV]"
    )


hamiltonian = system.full_hamiltonian()

m3l = np.array([1,1,1,1]+[0,0,0,0]*(no_dots-1)).T/2.
m3l_ = np.array([1,1,-1,-1]+[0,0,0,0]*(no_dots-1)).T*1.j/2.
m3r = np.array([0,0,0,0]*(no_dots-1)+[1,1,-1,-1]).T*1.j/2.
m3r_ = np.array([0,0,0,0]*(no_dots-1)+[1,1,1,1]).T/2.

m1 = np.concatenate([np.array([1,1,-1,-1]), np.array([0,0,0,0]*(no_dots-2)), np.array([1,1,1,1])]).T/2.
m2 = np.concatenate([np.array([1,1,-1,-1])*-1., np.array([0,0,0,0]*(no_dots-2)), np.array([1,1,1,1])]).T/2.

points = []
majoranas = []
for v in np.linspace(-1.5/AtomicUnits.Eh, 1.5/AtomicUnits.Eh, num=101):
    system.set_parameter("mu", np.ones(system.parameters.no_dots)*v)
    hamiltonian = system.full_hamiltonian()
    eigs, eigvs = eigh(hamiltonian, eigvals_only=False)
    ms = []
    for ie, eig in enumerate(eigs):
        ml = np.conj(eigvs[:,ie])@m1
        mr = np.conj(eigvs[:,ie])@m2
        zm = np.exp(-np.abs(eig)/(0.1/AtomicUnits.Eh))
        points.append([v, eig+ie*0.005/AtomicUnits.Eh, (np.abs(ml)-np.abs(mr))*zm])
        #points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=0).reshape(2,2).sum(axis=1)[0]])
        #points.append([v, eig+ie*0.005/au.Eh, (np.abs(eigvs[:,ie])**2).reshape(-1,4).sum(axis=1)[0]*2])
        ms.append(np.abs(ml-mr)*zm)  # mode projection on left - right majoranas
    ms = np.array(ms)[np.abs(eigs).argsort()]  # sort MZM_i using |E_i|
    majoranas.append([v, np.amax([0., ms[0]+ms[1]-ms[2:].sum()])])  # max(0, MZM_1+MZM_2 - all_others), everything is weighed by exp(-E_i/E_0)

points = np.array(points)
majoranas = np.array(majoranas)

fig, ax = plt.subplots()
ax.set_xlabel("v_gate")
ax.set_ylabel("energy")
ax.set_ylim([-1.,1.])
cc = ax.scatter(x=points[:,0]*AtomicUnits.Eh, y=points[:,1]*AtomicUnits.Eh, c=points[:,2], s=1., cmap='coolwarm')
ax.plot(majoranas[:,0]*AtomicUnits.Eh, majoranas[:,1]/2., c='orange')
cbar = fig.colorbar(cc, ax=ax)
cbar.set_label(r'majoranization')
plt.savefig(f'{test_dir}/rashba_3QD.png', dpi=200)
plt.close()
