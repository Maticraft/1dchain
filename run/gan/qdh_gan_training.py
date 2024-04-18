import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from src.data_utils import HamiltionianDataset, calculate_mean_and_std, Denormalize
from src.hamiltonian.helical_ladder import  DEFAULT_PARAMS, SpinLadder
from src.hamiltonian.quantum_dots_chain import QuantumDotsHamiltonian, QuantumDotsHamiltonianParameters, DefaultParameters, AtomicUnits
from src.models.gan import Generator
from src.models.distribution_preserving_autoencoder import DistributionPreservingEncoder, DistributionPreservingHamiltonianGenerator
from src.models.hamiltonian_generator import QuantumDotsHamiltonianGenerator
from src.models.gan import train_gan
from src.models.files import save_gan_params, save_gan, save_data_list, get_full_model_config, load_gan_submodel_state_dict, load_model, load_latent_distribution, save_latent_distribution, load_covariance_matrix, save_covariance_matrix
from src.models.gan import Discriminator, EigvalsDiscriminator
from src.plots import plot_convergence, plot_test_matrices, plot_test_eigvals, plot_matrix, plot_generator_eigvals

from src.models.positional_autoencoder import PositionalDecoder, PositionalEncoder

# Paths
data_path = './data/quantum_dots/7dots2levels_large'
data_mean_std_path = f'{data_path}/mean_std.pkl'
save_dir = './gan/quantum_dots/7dots2levels_large'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_latent_ep_{}'

original_autoencoder_path = './autoencoder/quantum_dots/7dots2levels_large/100/ditribution_preserving_autoencoder'
original_autoencoder_epoch = 20
distribution_path = os.path.join(original_autoencoder_path, distribution_dir_name.format(original_autoencoder_epoch))


# Reference eigvals plot params
eigvals_sub_dir = 'eigvals'
eigvals_gen_plot_name = 'eigvals_spectre.png'
x_axis = 'mu'
x_values = np.linspace(-1.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-1., 1.)
vscale = 1/AtomicUnits.Eh

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model name
model_name = 'QDH-WGAN-eigvals-distribution_preserving_autoencoder'

# Params
params = {
    'epochs': 200,
    'batch_size': 64,
    'N': 14,
    'in_channels': 10,
    'block_size': 4,
    'representation_dim': 100,
    'strategy': 'eigvals-discriminator-wgan',
    'gp_weight': 1.e-4,
    'discriminator_iters': 2,
    'generator_iters': 1,
    'start_training_mode': 'discriminator',
    'data_label': None,
    'use_feature_matching': False,
    'feature_matching_weight': 0.1,
}

# Architecture
discriminator_params = {
    'on_site_real_block_pairs': ['z1', 'zx', 'zz', 'iyiy'],
    'on_site_imag_block_pairs': ['1iy', 'xiy'],
    'interaction_real_block_pairs': ['z1', '1iy'],
    'interaction_imag_block_pairs': ['z1', '1z', '1x'],
    'seq_channels_num': 64,
    'enc_depth': 5,
    'enc_hidden_size': 256,
    'activation': 'leaky_relu',
    'seq_freq_enc_depth': 4,
    'seq_freq_enc_hidden_size': 128,
    'seq_enc_depth': 4,
    'seq_enc_hidden_size': 128,
    'lr': 1.e-5,
}

generator_params = {
    'on_site_real_block_pairs': ['z1', 'zx', 'zz', 'iyiy'],
    'on_site_imag_block_pairs': ['1iy', 'xiy'],
    'interaction_real_block_pairs': ['z1', '1iy'],
    'interaction_imag_block_pairs': ['z1', '1z', '1x'],
    'dec_depth': 4,
    'dec_hidden_size': 128,
    'seq_dec_depth': 4,
    'seq_dec_hidden_size': 128,
    'seq_channels_num': 64,
    'activation': 'leaky_relu',
    'lr': 1.e-4,
    'skip_noise_converter': True,
    'training_switch_loss_ratio': 1.2,
    'nn_in_features_split_index': 32,
}


# Set the root dir
root_dir = os.path.join(save_dir, f'{params["representation_dim"]}', model_name)
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

loss_path = os.path.join(root_dir, loss_file)
convergence_path = os.path.join(root_dir, convergence_file)

eigvals_sub_path = os.path.join(root_dir, eigvals_sub_dir)     
if not os.path.isdir(eigvals_sub_path):
    os.makedirs(eigvals_sub_path)

save_gan_params(params, generator_params, discriminator_params, root_dir)

# Try to load data statistics
try:
    with open(data_mean_std_path, 'rb') as f:
        mean, std = pickle.load(f)
except:
    data = HamiltionianDataset(data_path, label_idx=(3, 4), format='csr', threshold=0.15)
    data_loader = DataLoader(data, params['batch_size'])
    mean, std = calculate_mean_and_std(data_loader, device=device)
    with open(data_mean_std_path, 'wb') as f:
        pickle.dump((mean, std), f)

print('Data mean:', mean)
print('Data std:', std)

data = HamiltionianDataset(data_path, label_idx=(3, 4), format='csr', threshold=0.15, normalization_mean=mean, normalization_std=std)
train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])

generator_config = get_full_model_config(params, generator_params)
generator = Generator(DistributionPreservingHamiltonianGenerator, generator_config)
load_gan_submodel_state_dict(original_autoencoder_path, original_autoencoder_epoch, generator)

discriminator_config = get_full_model_config(params, discriminator_params)
discriminator = EigvalsDiscriminator(DistributionPreservingEncoder, discriminator_config)
load_gan_submodel_state_dict(original_autoencoder_path, original_autoencoder_epoch, discriminator)

print(generator)
print(discriminator)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_params['lr'])
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_params['lr'])

init_distribution = load_latent_distribution(distribution_path)
cov_matrix = load_covariance_matrix(distribution_path)

save_latent_distribution(init_distribution, root_dir)
save_covariance_matrix(cov_matrix, root_dir)

save_data_list(['Epoch', 'Generator loss', 'Discriminator loss'], loss_path, mode='w')

training_mode = params['start_training_mode']

for epoch in range(1, params['epochs'] + 1):
    gen_loss, disc_loss, training_mode = train_gan(
        generator,
        discriminator,
        train_loader,
        epoch,
        device,
        generator_optimizer,
        discriminator_optimizer,
        init_distribution,
        data_label=params['data_label'],
        strategy=params['strategy'],
        gradient_penalty_weight=params['gp_weight'],
        discriminator_repeats=params['discriminator_iters'],
        generator_repeats=params['generator_iters'],
        training_switch_loss_ratio=generator_params['training_switch_loss_ratio'],
        start_training_mode=training_mode,
        use_majoranas_feature_matching=params['use_feature_matching'],
        feature_matching_loss_weight=params['feature_matching_weight'],
    )
    save_gan(generator, discriminator, root_dir, epoch)
    save_data_list([epoch, gen_loss, disc_loss], loss_path)

    # Plot sample hamiltonian
    test_matrix_path = os.path.join(root_dir, f'test_{epoch}')
    generator.to(device)
    os.makedirs(test_matrix_path, exist_ok=True)
    num_states = 5
    for i in range(num_states):
        generator.eval()
        z = generator.get_noise(1, device=device, noise_type='custom', mean=init_distribution[0], std=init_distribution[1], covariance_matrix=cov_matrix)
        output = generator(z)

        # denormalize before plotting
        denormalization = Denormalize(mean, std)
        matrix = denormalization(output).detach().cpu().numpy()[0]
        
        plot_matrix(matrix[0], os.path.join(test_matrix_path, f"random_hamiltonian_real_{i}.png"), vmin=-vscale, vmax=vscale)
        plot_matrix(matrix[1], os.path.join(test_matrix_path, f"random_hamiltonian_imag_{i}.png"), vmin=-vscale, vmax=vscale)

    eigvals_gen_plot_path = os.path.join(test_matrix_path, eigvals_gen_plot_name)
    plot_generator_eigvals(generator, 5, eigvals_gen_plot_path, noise_type='custom', ylim=ylim, mean=init_distribution[0], std=init_distribution[1], covariance_matrix=cov_matrix, xnorm=xnorm, ynorm=ynorm, normalization_mean=mean, normalization_std=std)
   
plot_convergence(loss_path, convergence_path, read_label=True)
