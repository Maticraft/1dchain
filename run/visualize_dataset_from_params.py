import os
import pickle

import numpy as np
from torch.utils.data import DataLoader
import torch

from src.data.datasets import HamiltionianDataset, HamiltonianFromParametersDataset
from src.hamiltonian import quantum_dots_chain as qd_chain
from src.data.utils import calculate_mean_and_std
from src.hamiltonian.hamiltonian_torch_handlers import HamiltonianParams
from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.hamiltonian.quantum_dots_chain import QuantumDotsHamiltonian
from src.models.files import load_general_params, load_ae_model
from src.plots import plot_dataset_from_params_samples, plot_dataset_continous_samples
from src.models.positional_autoencoder import PositionalEncoder
from src.models.hamiltonian_generator import HamiltonianGeneratorV2, QuantumDotsHamiltonianGenerator

# model_dir = './autoencoder/quantum_dots/3dots1level/100/pos_encoder_qdh_generator'
epoch = 200
mzm_threshold = 0.02
num_samples = 10
num_hamiltonians = 2
vscale = 1/AtomicUnits.Eh
xnorm = 1/AtomicUnits.Eh
ynorm = 1/AtomicUnits.Eh

# Paths
data_path = './data/quantum_dots/simple_fixed_t_3dots1level_majoranization0_0_mzm_gap0_25'
data_mean_std_path = f'{data_path}/mean_std.pkl'

# test_dir_name = 'tests_subspace_{}_latent_ep{}'
test_dir_name = 'samples'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

hamiltonian_params=HamiltonianParams(
    inter_site_real_params=("t", "t_phase_diag", "t_phase_antidiag"),
    inter_site_imag_params=(()),
)

conductance_config = {
    'cmap_list': [
    {
        'cmap2': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'b_range': (-.5/AtomicUnits.Eh, .5/AtomicUnits.Eh),
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

data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, conductance_config=conductance_config)


save_path = os.path.join(data_path, test_dir_name)
os.makedirs(save_path, exist_ok=True)

# Load model
# params = load_general_params(model_dir)
# encoder, decoder = load_ae_model(model_dir, epoch, PositionalEncoder, QuantumDotsHamiltonianGenerator)

save_path_org_rep = os.path.join(save_path, 'majoranization_with_cmap')
os.makedirs(save_path_org_rep, exist_ok=True)
ids = np.arange(num_samples)
plot_dataset_from_params_samples(data, save_path_org_rep, num_samples=num_samples, n_dots=3, device=device, ylim=(-2.5, 2.5), vmin=-vscale, vmax=vscale, xnorm=xnorm, ynorm=ynorm, ids=ids, threshold=qd_chain.MZM_THRESHOLD, polaxis='y', string_num=1, cmap_config=conductance_config, mu_range=(-1./AtomicUnits.Eh, 1./AtomicUnits.Eh), majoranization=True, hamiltonian_params=hamiltonian_params)
