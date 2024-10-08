import os
import pickle

import numpy as np
from torch.utils.data import DataLoader
import torch

from src.data_utils import HamiltionianDataset, calculate_mean_and_std
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.hamiltonian.quantum_dots_chain import AtomicUnits
from src.models.files import load_general_params, load_ae_model
from src.plots import plot_dataset_samples, plot_dataset_continous_samples
from src.models.positional_autoencoder import PositionalEncoder
from src.models.hamiltonian_generator import HamiltonianGeneratorV2, QuantumDotsHamiltonianGenerator

model_dir = './autoencoder/quantum_dots/7dots2levels_defaults/100/pos_encoder_qdh_generator'
epoch = 200
mzm_threshold = 0.02
num_samples = 10
num_hamiltonians = 2
vscale = 1/AtomicUnits.Eh
xnorm = 1/AtomicUnits.Eh
ynorm = 1/AtomicUnits.Eh

# Paths
data_path = './data/quantum_dots/7dots2levels_fixed_balanced'
data_mean_std_path = f'{data_path}/mean_std.pkl'

# test_dir_name = 'tests_subspace_{}_latent_ep{}'
test_dir_name = 'samples'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_idx = (3, 4, 7)

try:
    with open(data_mean_std_path, 'rb') as f:
        mean, std = pickle.load(f)
except:
    data = HamiltionianDataset(data_path, label_idx=(3, 4), format='csr', threshold=mzm_threshold)
    data_loader = DataLoader(data, batch_size=64)
    mean, std = calculate_mean_and_std(data_loader, device=device)
    with open(data_mean_std_path, 'wb') as f:
        pickle.dump((mean, std), f)

save_path = os.path.join(data_path, test_dir_name)
os.makedirs(save_path, exist_ok=True)

# Load model
# params = load_general_params(model_dir)
# encoder, decoder = load_ae_model(model_dir, epoch, PositionalEncoder, QuantumDotsHamiltonianGenerator)

save_path_org_rep = os.path.join(save_path, 'original_representation')
os.makedirs(save_path_org_rep, exist_ok=True)
data = HamiltionianDataset(data_path, data_limit=100, label_idx=label_idx, eig_decomposition=False, format='csr', threshold=mzm_threshold)
ids = np.random.choice(len(data), num_samples, replace=False)
plot_dataset_samples(data, save_path_org_rep, num_samples=num_samples, plot_reconstructed_eigvals=False,  device=device, ylim=(-2.5, 2.5), vmin=-vscale, vmax=vscale, xnorm=xnorm, ynorm=ynorm, normalization_mean=mean, normalization_std=std, ids=ids)
ids2 = np.random.choice(len(data), num_samples*num_hamiltonians, replace=False)
plot_dataset_continous_samples(data, save_path_org_rep, num_samples=num_samples, ylim=(-2.5, 2.5), xnorm=xnorm, ynorm=ynorm, normalization_mean=mean, normalization_std=std, ids=ids2)

data_majorana_rep = HamiltionianDataset(data_path, data_limit=100, label_idx=label_idx, eig_decomposition=False, format='csr', threshold=mzm_threshold, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
save_path_majorana_rep = os.path.join(save_path, 'majorana_representation')
os.makedirs(save_path_majorana_rep, exist_ok=True)
plot_dataset_samples(data_majorana_rep, save_path_majorana_rep, num_samples=num_samples, plot_reconstructed_eigvals=False,  device=device, ylim=(-2.5, 2.5), vmin=-vscale, vmax=vscale, xnorm=xnorm, ynorm=ynorm, normalization_mean=mean, normalization_std=std, ids=ids)
plot_dataset_continous_samples(data_majorana_rep, save_path_majorana_rep, num_samples=num_samples, ylim=(-2.5, 2.5), xnorm=xnorm, ynorm=ynorm, normalization_mean=mean, normalization_std=std, ids=ids2)