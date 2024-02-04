import os

from torch.utils.data import DataLoader
import torch

from src.data_utils import HamiltionianDataset
from src.models.files import load_general_params, load_ae_model
from src.plots import plot_dataset_samples, plot_dataset_continous_samples
from src.models.positional_autoencoder import PositionalEncoder
from src.models.hamiltonian_generator import HamiltonianGeneratorV2

model_dir = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPGOnlyMajoranas/100/twice_pretrained_pos_encoder_hamiltonian_generator_v2_varying_potential_tf'
epoch = 40
mzm_threshold = 0.02
num_samples = 10

# Paths
data_path = './data/spin_ladder/70_2_RedDistSimplePeriodicPGSeparatedMajoranas'

# test_dir_name = 'tests_subspace_{}_latent_ep{}'
test_dir_name = 'tests_dataset_samples_{}'
dim_red_plot_file_name = 'tsne_class_{}.png'
latent_space_plot_name = 'latent_space.png'
correlation_matrix_name = 'covariance_matrix.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
label_idx = (3, 4, 7)

save_path = os.path.join(data_path, test_dir_name.format(epoch))
os.makedirs(save_path, exist_ok=True)

# Load model
params = load_general_params(model_dir)
encoder, decoder = load_ae_model(model_dir, epoch, PositionalEncoder, HamiltonianGeneratorV2)

data = HamiltionianDataset(data_path, data_limit=100, label_idx=label_idx, eig_decomposition=False, format='csr', threshold=mzm_threshold)
plot_dataset_samples(data, save_path, num_samples=num_samples, plot_reconstructed_eigvals=False, encoder=encoder, decoder=decoder, device=device, ylim=(-0.5, 0.5))
plot_dataset_continous_samples(data, save_path, num_samples=num_samples, ylim=(-0.5, 0.5))