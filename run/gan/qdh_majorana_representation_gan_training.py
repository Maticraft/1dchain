import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader, Subset, RandomSampler
import torch

from src.data.datasets import HamiltionianDataset
from src.data.utils import calculate_mean_and_std, Denormalize
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.hamiltonian.quantum_dots_chain import AtomicUnits
from src.models.gan import Generator
from src.models.autoencoder import Encoder, Decoder
from src.models.majorana_representation_generator import MajoranaRepresentationHamiltonianGenerator, BaselineHiddenRepresentationGenerator
from src.models.gan import train_gan
from src.models.files import save_gan_params, save_gan, save_data_list, get_full_model_config, load_latent_distribution, save_latent_distribution, load_covariance_matrix, save_covariance_matrix
from src.models.gan import Discriminator
from src.plots import plot_convergence, plot_matrix, plot_generator_eigvals


# Paths
data_path = './data/quantum_dots/7dots2levels_fixed_balanced'
data_mean_std_path = f'{data_path}/mean_std.pkl'
save_dir = './gan/quantum_dots/7dots2levels_fixed_balanced'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_majoranas_latent_ep_{}'

# Reference eigvals plot params
eigvals_sub_dir = 'eigvals'
eigvals_gen_plot_name = 'eigvals_spectre.png'
x_axis = 'mu'
x_values = np.linspace(-1.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-5., 5.)
vscale = 1/AtomicUnits.Eh

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model name
model_name = 'QDH-majorana_representation_generator_simple_convs_baseline_wgan_overfit_gen_iters_10_disc_lr_1e-4_gen_lr_1e-4'

# Params
params = {
    'epochs': 1000,
    'batch_size': 4,
    'N': 14,
    'in_channels': 2,
    'block_size': 4,
    'representation_dim': 100,
    'strategy': 'wgan-gp',
    'gp_weight': 0.,
    'discriminator_iters': 1,
    'generator_iters': 10,
    'start_training_mode': 'discriminator',
    'data_label': None,
    'use_feature_matching': False,
    'feature_matching_weight': 1.,
    'relative_noise_strength': 0.
}

# Architecture
discriminator_params = {
    'kernel_size': (1, 3),
    'kernel_size1': (4, 4),
    'stride': (1, 1),
    'stride1': 4,
    'dilation': 1,
    'dilation1': 1,
    'fc_num': 1,
    'conv_num': 1,
    'kernel_num': 64,
    'kernel_num1': 64,
    'hidden_size': 64,
    'activation': 'leaky_relu',
    'use_strips': False,
    'lr': 1.e-5,
}

hidden_representation_generator_params = {
    'num_hidden_mlps': 10,
    'layers_num': 5,
    'input_size': 100,
    'hidden_size': 64,
    'output_size': 14,
    'activation': 'relu',
    'final_activation': 'none'
}
hidden_representation_generator = BaselineHiddenRepresentationGenerator(**hidden_representation_generator_params)
generator_params = {
    'hidden_representation_generator': hidden_representation_generator,
    'min_inter_site_interaction_range': 2,
    'max_inter_site_interaction_range': 3,
    'lr': 5.e-4,
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
    data = HamiltionianDataset(data_path, label_idx=(3, 4), format='csr', threshold=0.15, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
    data_loader = DataLoader(data, params['batch_size'])
    mean, std = calculate_mean_and_std(data_loader, device=device)
    with open(data_mean_std_path, 'wb') as f:
        pickle.dump((mean, std), f)

print('Data mean:', mean)
print('Data std:', std)

data = HamiltionianDataset(data_path, data_limit=1, label_idx=(3, 4), format='csr', threshold=0.15, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
# data = Subset(data, range(len(data) // 2, len(data)))
# train_size = int(0.99*len(data))
# test_size = len(data) - train_size

# train_data, test_data = random_split(data, [train_size, test_size])
# train_loader = DataLoader(train_data, params['batch_size'])
# test_loader = DataLoader(test_data, params['batch_size'])
sampler = RandomSampler(data, replacement=True, num_samples=20)
train_loader = DataLoader(data, params['batch_size'], sampler=sampler)

generator_config = get_full_model_config(params, generator_params)
generator = Generator(MajoranaRepresentationHamiltonianGenerator, generator_config)

discriminator_config = get_full_model_config(params, discriminator_params)
discriminator = Discriminator(Encoder, discriminator_config)


print(generator)
print(discriminator)

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=generator_params['lr'])
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=discriminator_params['lr'])


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
        training_switch_loss_ratio=generator_params['training_switch_loss_ratio'],
        start_training_mode=training_mode,
        use_majoranas_feature_matching=params['use_feature_matching'],
        feature_matching_loss_weight=params['feature_matching_weight'],
        relative_noise_strength=params['relative_noise_strength'],
        polarization_representation_mapping=RepresentationMapping.majorana_plus_minus_up_down
    )
    save_gan(generator, discriminator, root_dir, epoch)
    save_data_list([epoch, gen_loss, disc_loss], loss_path)

    # Plot sample hamiltonian
    test_matrix_path = os.path.join(root_dir, 'tests', f'test_{epoch}')
    generator.to(device)
    denormalization = Denormalize(mean, std)
    data_sample = next(iter(train_loader))[0][0]
    matrix = denormalization(data_sample).detach().cpu().numpy()[0]
    os.makedirs(test_matrix_path, exist_ok=True)
    plot_matrix(matrix[0], os.path.join(test_matrix_path, f"data_hamiltonian_real.png"), vmin=-vscale, vmax=vscale)
    plot_matrix(matrix[1], os.path.join(test_matrix_path, f"data_hamiltonian_imag.png"), vmin=-vscale, vmax=vscale)
    num_states = 5
    for i in range(num_states):
        generator.eval()
        z = generator.get_noise(1, device=device, noise_type='gaussian')
        output = generator(z)

        # denormalize before plotting
        matrix = denormalization(output).detach().cpu().numpy()[0]
        
        plot_matrix(matrix[0], os.path.join(test_matrix_path, f"random_hamiltonian_real_{i}.png"), vmin=-vscale, vmax=vscale)
        plot_matrix(matrix[1], os.path.join(test_matrix_path, f"random_hamiltonian_imag_{i}.png"), vmin=-vscale, vmax=vscale)

    eigvals_gen_plot_path = os.path.join(test_matrix_path, eigvals_gen_plot_name)
    plot_generator_eigvals(generator, 5, eigvals_gen_plot_path, noise_type='gaussian', ylim=ylim, xnorm=xnorm, ynorm=ynorm, normalization_mean=mean, normalization_std=std)
   
plot_convergence(loss_path, convergence_path, read_label=True)
