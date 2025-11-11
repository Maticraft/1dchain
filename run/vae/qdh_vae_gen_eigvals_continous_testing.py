import os
import pickle

from tqdm import tqdm
import numpy as np

from src.data.datasets import HamiltionianDataset
from src.data.utils import Denormalize
from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.quantum_dots_chain import MZM_THRESHOLD, DefaultParameters, QuantumDotsHamiltonian, QuantumDotsHamiltonianParameters
from src.hamiltonian.utils import plot_majorana_polarization, plot_eigvals_levels
from src.models.gan import Generator
from src.models.files import load_ae_model, load_latent_distribution, load_autoencoder_params, get_full_model_config, load_gan_submodel_state_dict, load_covariance_matrix, load_positional_autoencoder
from src.plots import plot_generator_eigvals, plot_matrix, plot_generator_noisy_sample_eigvals
from src.models.distribution_preserving_autoencoder import DistributionPreservingHamiltonianGenerator
from src.models.hamiltonian_generator import HamiltonianGenerator
from src.hamiltonian.torch_hamiltonian import TorchHamiltonian
from src.models.utils import calculate_pca
from src.models.distribution_preserving_autoencoder import DistributionPreservingHamiltonianGenerator, VariationalDistributionPreservingEncoder

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize


# Model params
ae_dir = './vae/quantum_dots/7dots2levels_large_balanced/100/majoranas_distribution_preserving_autoencoder_kl_01_weighting'
test_dir_name = 'majoranas_generation_tests_from_mean_10p_ep_{}'
latent_distrib_dir = 'tests_majoranas_latent_ep_{}'
polarization_sub_dir = 'polarization_{}'
gen_epoch = 20

eigvals_gen_plot_name = 'eigvals_spectre_generator_{}.png'
noisy_eigvals_plot_name = 'noisy_eigvals_spectre_generator_{}.png'
hamiltonian_plot_name = 'hamiltonian_{}.png'

data_path = './data/quantum_dots/7dots2levels_large_balanced'
data_mean_std_path = f'{data_path}/mean_std.pkl'
batch_size = 128


# Eigvals plot params
num_states = 3
num_plots = 10
x_axis = 'mu'
x_values = np.linspace(-1.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-1., 1.)
vscale = 1/AtomicUnits.Eh

eps = 0.1


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------------------------------------------------------------------------------------
test_sub_path = os.path.join(ae_dir, test_dir_name.format(gen_epoch))    
if not os.path.isdir(test_sub_path):
    os.makedirs(test_sub_path)

params, encoder_params, decoder_params = load_autoencoder_params(ae_dir, VariationalDistributionPreservingEncoder, DistributionPreservingHamiltonianGenerator)
generator_config = get_full_model_config(params, decoder_params)
generator_config['skip_noise_converter'] = True
generator_config['nn_in_features_split_index'] = 32
generator = Generator(DistributionPreservingHamiltonianGenerator, generator_config)
load_gan_submodel_state_dict(ae_dir, gen_epoch, generator)
generator.eval()

latent_space_sub_path = os.path.join(ae_dir, latent_distrib_dir.format(gen_epoch))
mean, std = load_latent_distribution(latent_space_sub_path)
cov_matrix = load_covariance_matrix(latent_space_sub_path)

# Enforce positive definiteness for cov_matrix
# eigs = torch.amin(torch.linalg.eigvalsh(cov_matrix))
# if eigs < 0.: cov_matrix -= torch.eye(cov_matrix.shape[-1])*eigs  


with open(data_mean_std_path, 'rb') as f:
    normalization_mean, normalization_std = pickle.load(f)


# -----------------
encoder, _ = load_ae_model(ae_dir, gen_epoch, VariationalDistributionPreservingEncoder, DistributionPreservingHamiltonianGenerator)
encoder.eval()
default_params = DefaultParameters()
default_params.mu_default = -0.5/AtomicUnits.Eh

# --------------------------

for i in tqdm(range(num_plots), desc='Plotting generator eigvals'):
    # ------
    # Real sample
    parameters = QuantumDotsHamiltonianParameters(no_dots=7, no_levels=2, default_parameters=default_params)
    H = QuantumDotsHamiltonian(parameters).get_hamiltonian()
    H_torch = torch.from_numpy(H)
    H_torch = torch.stack((H_torch.real, H_torch.imag), dim= 0)
    normalization = Normalize(normalization_mean, normalization_std)
    H_torch = normalization(H_torch)
    H_torch = H_torch.unsqueeze(0).float()
    real_latent_vec = encoder(H_torch)
    # ----------------
    eigvals_gen_plot_path = os.path.join(test_sub_path, eigvals_gen_plot_name.format(f'{i}'))
    plot_generator_eigvals(generator, num_states, eigvals_gen_plot_path, noise_type='covariance', ylim=ylim, mean=mean, covariance=cov_matrix, std=std, normalization_mean=normalization_mean, normalization_std=normalization_std, ynorm=ynorm, real_sample=real_latent_vec, noise_strength=eps)

    noisy_eigvals_plot_path = os.path.join(test_sub_path, noisy_eigvals_plot_name.format(f'{i}'))
    plot_generator_noisy_sample_eigvals(generator, noisy_eigvals_plot_path, real_latent_vec, noise_type='covariance', ylim=ylim, mean=mean, std=std, covariance=cov_matrix, normalization_mean=normalization_mean, normalization_std=normalization_std, ynorm=ynorm, eps=eps)

    input_noise = generator.get_noise(1, device='cpu', noise_type='covariance', mean=mean, std=std, covariance=cov_matrix)
    input_noise = (1 - eps)*real_latent_vec + eps*input_noise
    H = generator(input_noise)
    
    denormalization = Denormalize(normalization_mean, normalization_std)
    H = denormalization(H).squeeze()
    H = TorchHamiltonian.from_2channel_tensor(H)
    
    matrix = H.get_hamiltonian()

    polarization_sub_path = os.path.join(test_sub_path, polarization_sub_dir.format(i))
    hamiltonian_real_path = os.path.join(test_sub_path, hamiltonian_plot_name.format(f'{i}_real'))
    hamiltonian_imag_path = os.path.join(test_sub_path, hamiltonian_plot_name.format(f'{i}_imag'))
    plot_matrix(np.real(matrix), hamiltonian_real_path, vmin=-vscale, vmax=vscale)
    plot_matrix(np.imag(matrix), hamiltonian_imag_path, vmin=-vscale, vmax=vscale)
    plot_majorana_polarization(H, polarization_sub_path, threshold = MZM_THRESHOLD, string_num=1)
    plot_eigvals_levels(H, os.path.join(test_sub_path, 'eigvals_levels_{}.png'.format(i)), ylim=ylim, ynorm=ynorm)
