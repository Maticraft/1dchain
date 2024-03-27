import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from src.data_utils import HamiltionianDataset, calculate_mean_and_std
from src.hamiltonian.helical_ladder import  DEFAULT_PARAMS, SpinLadder
from src.models.gan import Generator
from src.models.hamiltonian_generator import QuantumDotsHamiltonianGenerator
from src.models.gan import train_gan
from src.models.files import save_gan_params, save_gan, save_data_list, get_full_model_config, load_gan_submodel_state_dict, load_model, load_latent_distribution, save_latent_distribution, load_covariance_matrix, save_covariance_matrix
from src.models.gan import Discriminator
from src.plots import plot_convergence, plot_test_matrices, plot_test_eigvals, plot_matrix, plot_generator_eigvals

from src.models.autoencoder import Decoder, Encoder

# Paths
data_path = './data/quantum_dots/7dots2levels_defaults'
data_mean_std_path = f'{data_path}/mean_std.pkl'
save_dir = './gan/quantum_dots/7dots2levels_defaults'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_latent_majoranas_ep_{}'

# Reference eigvals plot params
eigvals_sub_dir = 'eigvals'
x_axis = 'q'
x_values = np.arange(0., np.pi, 0.1)
xnorm = np.pi
ylim = (-0.5, 0.5)
eigvals_gen_plot_name = 'eigvals_spectre_generator_{}.png'

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model name
model_name = 'Simple_WGAN-GP-strips'

# Params
params = {
    'epochs': 1000,
    'batch_size': 64,
    'N': 14,
    'in_channels': 10,
    'block_size': 4,
    'representation_dim': 100,
    'strategy': 'wgan-gp',
    'gp_weight': 1.e-3,
    'discriminator_iters': 1,
    'generator_iters': 5,
    'start_training_mode': 'generator',
    'data_label': None,
    'use_feature_matching': False,
    'feature_matching_weight': 0.1,
}

encoder_params = {
    'kernel_size': (1, 3),
    'kernel_size1': (4, 4),
    'stride': (1, 1),
    'stride1': 4,
    'dilation': 1,
    'dilation1': 1,
    'fc_num': 4,
    'conv_num': 3,
    'kernel_num': 64,
    'kernel_num1': 64,
    'hidden_size': 512,
    'activation': 'leaky_relu',
    'use_strips': True,
    'lr': 1.e-4,

}

decoder_params = {
    'kernel_size': (1, 3),
    'kernel_size1': (4, 4),
    'stride': (1, 1),
    'stride1': 4,
    'dilation': 1,
    'dilation1': 1,
    'fc_num': 4,
    'conv_num': 3,
    'kernel_num': 64,
    'kernel_num1': 64,
    'hidden_size': 512,
    'upsample_method': 'transpose',
    'scale_factor': 2, # does matter only for upsample_method 'nearest' or 'bilinear'
    'activation': 'leaky_relu',
    'use_strips': True,
    'lr': 1.e-4,
    'training_switch_loss_ratio': 1.2,
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

save_gan_params(params, decoder_params, encoder_params, root_dir)

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

generator_config = get_full_model_config(params, decoder_params)
generator = Generator(Decoder, generator_config)

discriminator_config = get_full_model_config(params, encoder_params)
discriminator = Discriminator(Encoder, discriminator_config)

print(generator)
print(discriminator)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=decoder_params['lr'])
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=encoder_params['lr'])

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
        data_label=params['data_label'],
        strategy=params['strategy'],
        gradient_penalty_weight=params['gp_weight'],
        discriminator_repeats=params['discriminator_iters'],
        generator_repeats=params['generator_iters'],
        training_switch_loss_ratio=decoder_params['training_switch_loss_ratio'],
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
        z = generator.get_noise(1, device=device, noise_type='hybrid')
        matrix = generator.nn(z).detach().cpu().numpy()[0]
        
        plot_matrix(matrix[0], os.path.join(test_matrix_path, f"random_hamiltonian_real_{i}.png"))
        plot_matrix(matrix[1], os.path.join(test_matrix_path, f"random_hamiltonian_imag_{i}.png"))

        eigvals_gen_plot_path = os.path.join(test_matrix_path, eigvals_gen_plot_name.format(i))
    plot_generator_eigvals(generator, 5, eigvals_gen_plot_path, noise_type='hybrid', ylim=ylim)
   
plot_convergence(loss_path, convergence_path, read_label=True)
