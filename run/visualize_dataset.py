import os
import pickle

from torch.utils.data import DataLoader
import torch

from src.data_utils import HamiltionianDataset
from src.hamiltonian.quantum_dots_chain import AtomicUnits
from src.models.files import load_general_params, load_ae_model
from src.plots import plot_dataset_samples, plot_dataset_continous_samples
from src.models.positional_autoencoder import PositionalEncoder
from src.models.hamiltonian_generator import HamiltonianGeneratorV2, QuantumDotsHamiltonianGenerator

model_dir = './autoencoder/quantum_dots/7dots2levels_defaults/100/pos_encoder_qdh_generator'
epoch = 200
mzm_threshold = 0.02
num_samples = 10
vscale = 1/AtomicUnits.Eh
xnorm = 1/AtomicUnits.Eh
ynorm = 1/AtomicUnits.Eh

# Paths
data_path = './data/quantum_dots/7dots2levels_simplified'
data_mean_std_path = f'{data_path}/mean_std.pkl'

# test_dir_name = 'tests_subspace_{}_latent_ep{}'
test_dir_name = 'tests_dataset_samples'
dim_red_plot_file_name = 'tsne_class_{}.png'
latent_space_plot_name = 'latent_space.png'
correlation_matrix_name = 'covariance_matrix.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_idx = (3, 4, 7)

with open(data_mean_std_path, 'rb') as f:
    mean, std = pickle.load(f)

save_path = os.path.join(data_path, test_dir_name)
os.makedirs(save_path, exist_ok=True)

# Load model
params = load_general_params(model_dir)
encoder, decoder = load_ae_model(model_dir, epoch, PositionalEncoder, QuantumDotsHamiltonianGenerator)

data = HamiltionianDataset(data_path, data_limit=100, label_idx=label_idx, eig_decomposition=False, format='csr', threshold=mzm_threshold)
plot_dataset_samples(data, save_path, num_samples=num_samples, plot_reconstructed_eigvals=False,  device=device, ylim=(-2.5, 2.5), encoder=encoder, decoder=decoder, vmin=-vscale, vmax=vscale, xnorm=xnorm, ynorm=ynorm, normalization_mean=mean, normalization_std=std)
plot_dataset_continous_samples(data, save_path, num_samples=num_samples, ylim=(-2.5, 2.5), xnorm=xnorm, ynorm=ynorm, normalization_mean=mean, normalization_std=std)