import os

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from src.data_utils import HamiltionianDataset
from src.hamiltonian.helical_ladder import  DEFAULT_PARAMS, SpinLadder
from src.models.gan import Generator
from src.models.hamiltonian_generator import HamiltonianGenerator, HamiltonianGeneratorV2
from src.models.gan import train_gan
from src.models.files import save_gan_params, save_gan, save_data_list, get_full_model_config, load_gan_submodel_state_dict, load_model, load_latent_distribution, save_latent_distribution, load_covariance_matrix, save_covariance_matrix
from src.models.gan import Discriminator
from src.plots import plot_convergence, plot_test_matrices, plot_test_eigvals, plot_matrix
from src.models.positional_autoencoder import PositionalDecoder, PositionalEncoder

# Paths
data_path = './data/spin_ladder/70_2_RedDistSimplePeriodicPGBalancedZM'
save_dir = './gan/spin_ladder/70_2_RedDistSimplePeriodicPGBalancedZM'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_latent_ep{}'

# Load state from pretrained autoencoder
original_autoencoder_path = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPGBalancedZM/100/twice_pretrained_pos_encoder_hamiltonian_generator_tf'
original_autoencoder_epoch = 22
distribution_path = os.path.join(original_autoencoder_path, distribution_dir_name.format(original_autoencoder_epoch))


# Reference eigvals plot params
eigvals_sub_dir = 'eigvals'
eigvals_plot_name = 'eigvals_spectre_autoencoder{}.png'
x_axis = 'q'
x_values = np.arange(0., np.pi, 0.1)
xnorm = np.pi
ylim = (-0.5, 0.5)

# Reference hamiltonian params
hamiltonian_sub_dir = 'hamiltonian'
hamiltonian_plot_name = 'hamiltonian_autoencoder{}.png'
hamiltonain_diff_plot_name = 'hamiltonian_diff{}.png'

# Model name
model_name = 'Hamiltonian_GAN_V2_fft_tf_dynamic_switch_no_noise_converter'

# Params
params = {
    'epochs': 40,
    'batch_size': 64,
    'N': 140,
    'in_channels': 10,
    'block_size': 4,
    'representation_dim': 100,
    'start_training_mode': 'discriminator',
    'data_label': 1
}

# Architecture
encoder_params = {
    'kernel_num': 64,
    'kernel_size': 4,
    'activation': 'leaky_relu',
    'freq_enc_depth': 4,
    'freq_enc_hidden_size': 128,
    'block_enc_depth': 4,
    'block_enc_hidden_size': 128,
    'padding_mode': 'zeros',
}


discriminator_params = {
    'kernel_num': 64,
    'kernel_size': 4,
    'activation': 'leaky_relu',
    'freq_enc_depth': 4,
    'freq_enc_hidden_size': 128,
    'block_enc_depth': 4,
    'block_enc_hidden_size': 128,
    'padding_mode': 'zeros',
    'lr': 1.e-6,
}

generator_params = {
    # "kernel_num": 64,
    "activation": "leaky_relu",
    "freq_dec_depth": 4,
    "freq_dec_hidden_size": 128,
    "block_dec_depth": 4,
    "block_dec_hidden_size": 128,
    "seq_dec_depth": 4,
    "seq_dec_hidden_size": 128,
    'lr': 1.e-5,
    'skip_noise_converter': True,
    'training_switch_loss_ratio': 1.2,
    'reduce_blocks': False,
    'seq_num': 32,
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

ham_sub_path = os.path.join(root_dir, hamiltonian_sub_dir)     
if not os.path.isdir(ham_sub_path):
    os.makedirs(ham_sub_path)

save_gan_params(params, generator_params, discriminator_params, root_dir)

data = HamiltionianDataset(data_path, label_idx=(3, 4), format='csr', threshold=0.15)

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])

generator_config = get_full_model_config(params, generator_params)
generator = Generator(HamiltonianGeneratorV2, generator_config)
# load_gan_submodel_state_dict(original_autoencoder_path, original_autoencoder_epoch, generator)

discriminator_config = get_full_model_config(params, discriminator_params)
discriminator = Discriminator(PositionalEncoder, discriminator_config)
load_gan_submodel_state_dict(original_autoencoder_path, original_autoencoder_epoch, discriminator)

encoder_config = get_full_model_config(params, encoder_params)
encoder = load_model(PositionalEncoder, encoder_config, original_autoencoder_path, original_autoencoder_epoch)

print(generator)
print(discriminator)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_params['lr'])
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_params['lr'])

init_distribution = load_latent_distribution(distribution_path)
cov_matrix = load_covariance_matrix(distribution_path)

save_latent_distribution(init_distribution, root_dir)
save_covariance_matrix(cov_matrix, root_dir)

save_data_list(['Epoch', 'Generator loss', 'Discriminator loss'], loss_path, mode='w')

training_mode = 'discriminator'

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
        cov_matrix=cov_matrix,
        training_switch_loss_ratio=generator_params['training_switch_loss_ratio'],
        start_training_mode=params['start_training_mode'],
        data_label=params['data_label']
    )
    save_gan(generator, discriminator, root_dir, epoch)
    save_data_list([epoch, gen_loss, disc_loss], loss_path)

    eigvals_path = os.path.join(eigvals_sub_path, eigvals_plot_name.format(f'_ep{epoch}'))
    plot_test_eigvals(SpinLadder, encoder, generator.nn, x_axis, x_values, DEFAULT_PARAMS, eigvals_path, device=device, xnorm=xnorm, ylim=ylim)
    ham_auto_path = os.path.join(ham_sub_path, hamiltonian_plot_name.format(f'_ep{epoch}' + '{}'))
    ham_diff_path = os.path.join(ham_sub_path, hamiltonain_diff_plot_name.format(f'_ep{epoch}'))
    plot_test_matrices(SpinLadder(**DEFAULT_PARAMS).get_hamiltonian(), encoder, generator.nn, save_path_rec=ham_auto_path, save_path_diff=ham_diff_path, device=device)

    # Plot sample hamiltonian
    test_matrix_path = os.path.join(root_dir, f'test_{epoch}')
    generator.to(device)
    os.makedirs(test_matrix_path, exist_ok=True)
    for i in range(10):
        generator.eval()
        z = generator.get_noise(1, device=device, noise_type='covariance', mean=init_distribution[0], covariance=cov_matrix)
        matrix = generator.nn(z).detach().cpu().numpy()[0]
        
        plot_matrix(matrix[0], os.path.join(test_matrix_path, f"random_hamiltonian_real_{i}.png"))
        plot_matrix(matrix[1], os.path.join(test_matrix_path, f"random_hamiltonian_imag_{i}.png"))
   
plot_convergence(loss_path, convergence_path, read_label=True)
