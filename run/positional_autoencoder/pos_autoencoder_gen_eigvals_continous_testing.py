import os
from tqdm import tqdm
import numpy as np

from src.data_utils import HamiltionianDataset
from src.majorana_utils import plot_majorana_polarization, plot_eigvals_levels
from src.models.gan import Generator
from src.models.files import load_generator, load_latent_distribution, load_autoencoder_params, get_full_model_config, load_gan_submodel_state_dict, load_covariance_matrix, load_positional_autoencoder
from src.plots import plot_generator_eigvals, plot_matrix
from src.models.positional_autoencoder import PositionalDecoder, PositionalEncoder
from src.torch_utils import TorchHamiltonian
from src.models.utils import calculate_pca

import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader



# Model params
ae_dir = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPG/100/classifier_bal_twice_pretrained_positional_autoencoder_fft_tf'
test_dir_name = 'generation_majoranas_015_ep{}'
latent_distrib_dir = 'tests_majoranas_015_latent_pca_ep{}'
polarization_sub_dir = 'polarization_{}'
gen_epoch = 4

eigvals_gen_plot_name = 'eigvals_spectre_generator_{}.png'
hamiltonian_plot_name = 'hamiltonian_{}.png'

data_path = './data/spin_ladder/70_2_RedDistSimplePeriodicPG'
batch_size = 128


# Eigvals plot params
num_states = 3
num_plots = 10
ylim = (-0.5, 0.5)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------------------------------------------------------------------------------------
test_sub_path = os.path.join(ae_dir, test_dir_name.format(gen_epoch))    
if not os.path.isdir(test_sub_path):
    os.makedirs(test_sub_path)

params, encoder_params, decoder_params = load_autoencoder_params(ae_dir, PositionalEncoder, PositionalDecoder)
generator_config = get_full_model_config(params, decoder_params)
generator_config['skip_noise_converter'] = True
generator = Generator(PositionalDecoder, **generator_config)
load_gan_submodel_state_dict(ae_dir, gen_epoch, generator)

latent_space_sub_path = os.path.join(ae_dir, latent_distrib_dir.format(gen_epoch))
mean, std = load_latent_distribution(latent_space_sub_path)
cov_matrix = load_covariance_matrix(latent_space_sub_path)

mean_freq = mean[:50]
mean_block = mean[50:]
cov_freq = cov_matrix[:50, :50]
cov_block = cov_matrix[50:, 50:]

mvn_freq = MultivariateNormal(mean_freq, covariance_matrix=cov_freq)
mvn_block = MultivariateNormal(mean_block, covariance_matrix=cov_block)

# for pca
# encoder, decoder = load_positional_autoencoder(ae_dir, gen_epoch)
# data = HamiltionianDataset(data_path, data_limit=10000, label_idx=(3, 4), eig_decomposition=False, format='csr', threshold=0.15)
# test_loader = DataLoader(data, batch_size)
# pca, pca_z, input_mean1 = calculate_pca(encoder, test_loader, device='cpu', label=1, n_components=5)
# print(pca.explained_variance_ratio_)
# input_mean2 = pca.inverse_transform(pca_z)

for i in tqdm(range(num_plots), desc='Plotting generator eigvals'):
    # mean1 = torch.from_numpy(input_mean1[i % len(input_mean1)]).float()
    # mean2 = torch.from_numpy(input_mean2[i % len(input_mean2)]).float()
    # std = torch.zeros_like(mean) + 0.1

    # eigvals_gen_plot_path1 = os.path.join(test_sub_path, eigvals_gen_plot_name.format(f'enc_{i}'))
    # eigvals_gen_plot_path2 = os.path.join(test_sub_path, eigvals_gen_plot_name.format(f'pca_{i}'))

    # plot_generator_eigvals(generator, num_states, eigvals_gen_plot_path1, noise_type='custom', ylim=ylim, mean=mean1, std=std)
    # plot_generator_eigvals(generator, num_states, eigvals_gen_plot_path2, noise_type='custom', ylim=ylim, mean=mean2, std=std)

    eigvals_gen_plot_path = os.path.join(test_sub_path, eigvals_gen_plot_name.format(f'{i}'))
    plot_generator_eigvals(generator, num_states, eigvals_gen_plot_path, noise_type='covariance', ylim=ylim, mean=mean, covariance=cov_matrix)

    # input_noise1 = generator.get_noise(1, device='cpu', noise_type='custom', mean=mean1, std=std)
    # input_noise2 = generator.get_noise(1, device='cpu', noise_type='custom', mean=mean2, std=std)

    input_noise = generator.get_noise(1, device='cpu', noise_type='covariance', mean=mean, covariance=cov_matrix)
    # input_noise_freq = mvn_freq.sample((1,))
    # input_noise_block = mvn_block.sample((1,))
    # input_noise = torch.cat((input_noise_freq, input_noise_block), dim=-1)

    H = generator(input_noise).squeeze()
    H = TorchHamiltonian.from_2channel_tensor(H)
    matrix = H.get_hamiltonian()

    polarization_sub_path = os.path.join(test_sub_path, polarization_sub_dir.format(i))
    hamiltonian_real_path = os.path.join(test_sub_path, hamiltonian_plot_name.format(f'{i}_real'))
    hamiltonian_imag_path = os.path.join(test_sub_path, hamiltonian_plot_name.format(f'{i}_imag'))
    plot_matrix(np.real(matrix), hamiltonian_real_path)
    plot_matrix(np.imag(matrix), hamiltonian_imag_path)
    plot_majorana_polarization(H, polarization_sub_path, threshold = 1.e-2, string_num=2)
    plot_eigvals_levels(H, os.path.join(test_sub_path, 'eigvals_levels_{}.png'.format(i)), ylim=ylim)

    # H2 = generator(input_noise2).squeeze()
    # H2 = TorchHamiltonian.from_2channel_tensor(H2)
    # matrix2 = H2.get_hamiltonian()
    # plot_eigvals_levels(H2, os.path.join(test_sub_path, 'eigvals_levels_pca_{}.png'.format(i)), ylim=ylim)
