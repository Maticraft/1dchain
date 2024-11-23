import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader, Subset, RandomSampler
import torch

from src.data_utils import HamiltionianDataset, calculate_mean_and_std, Denormalize
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.hamiltonian.quantum_dots_chain import AtomicUnits
from src.hamiltonian.hamiltonian_torch_handlers import ALL_PAIRS
from src.models.autoencoder import Encoder
from src.models.noise_generatiron import NoiseGenerator
from src.models.diffusion import DiffusionAutoencoder, train_diffusion_model, sample_ddpm
from src.models.positional_autoencoder import PositionalEncoder
from src.models.files import save_autoencoder_params, save_autoencoder, save_data_list, get_full_model_config, load_latent_distribution, save_latent_distribution, load_covariance_matrix, save_covariance_matrix
from src.models.majorana_representation_encoder import MajoranaRepresentationHamiltonianEncoder, BaselineHiddenRepresentationEncoder, SiteIndependentHiddenRepresentationEncoder, UNetLikeSiteIndependentHiddenRepresentationEncoder
from src.models.majorana_representation_generator import MajoranaRepresentationHamiltonianGenerator, BaselineHiddenRepresentationGenerator, SiteIndependentRepresentationGenerator, UNetLikeSiteIndependentHiddenRepresentationGenerator
from src.plots import plot_convergence, plot_matrix, plot_generator_eigvals
from src.hamiltonian.utils import plot_eigvals_levels
from src.torch_utils import TorchHamiltonian

# Paths
data_path = './data/quantum_dots/7dots2levels_fixed_balanced'
data_mean_std_path = f'{data_path}/mean_std.pkl'
save_dir = './diffusion/quantum_dots/7dots2levels_fixed_balanced'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_majoranas_latent_ep_{}'


# Reference eigvals plot params
eigvals_sub_dir = 'eigvals'
x_axis = 'mu'
x_values = np.linspace(-1.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-5., 5.)
vscale = 1/AtomicUnits.Eh

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model name
model_name = 'QDH-1lvl-no-interlevel_site_constant_majorana-unet-site-constant-deep6x256_lr1e-4_overfit'

# Params
params = {
    'epochs': 1000,
    'batch_size': 64,
    'N': 14,
    'block_size': 4,
    'representation_dim': 256,
    'strategy': 'no-discriminator',
    'data_label': None,
    'latent_shapes': (256, 256, 256, 256, 256, 256),
}

# Architecture
hidden_representation_encoder_params = {
    'unet_layers': 6,
    'seq_size': 14,
    'unet_hidden_size': 256,
    'hidden_representation_size': 256,
    'output_size': params['representation_dim'],
    'repeat_along_seq': True,
}
hidden_representation_encoder = UNetLikeSiteIndependentHiddenRepresentationEncoder(**hidden_representation_encoder_params)
encoder_params = {
    'hidden_representation_encoder': hidden_representation_encoder,
    'min_inter_site_interaction_range': 2,
    'max_inter_site_interaction_range': 3,
    'lr': 1.e-4,
}

hidden_representation_generator_params = {
    'hidden_representation_size': 256,
    'unet_layers': 4,
    'input_size': params['representation_dim'],
    'unet_hidden_size': 256,
    'seq_size': 14,
    'repeat_along_seq': True,
}
hidden_representation_generator = UNetLikeSiteIndependentHiddenRepresentationGenerator(**hidden_representation_generator_params)
generator_params = {
    'hidden_representation_generator': hidden_representation_generator,
    'min_inter_site_interaction_range': 2,
    'max_inter_site_interaction_range': 3,
    'lr': 1.e-4,
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

save_autoencoder_params(params, encoder_params, generator_params, root_dir)

# Try to load data statistics
try:
    with open(data_mean_std_path, 'rb') as f:
        mean, std = pickle.load(f)
except:
    data = HamiltionianDataset(data_path, label_idx=1, format='csr', threshold=0.4, gt_threshold=True, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
    data_loader = DataLoader(data, params['batch_size'])
    mean, std = calculate_mean_and_std(data_loader, device=device)
    with open(data_mean_std_path, 'wb') as f:
        pickle.dump((mean, std), f)

print('Data mean:', mean)
print('Data std:', std)

data = HamiltionianDataset(data_path, label_idx=1, format='csr', threshold=0.4, gt_threshold=True, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
train_data = Subset(data, range(len(data) // 2, len(data) // 2 + 1))

sampler = RandomSampler(train_data, replacement=True, num_samples=1000)
train_loader = DataLoader(train_data, params['batch_size'], sampler=sampler)

encoder = MajoranaRepresentationHamiltonianEncoder(**encoder_params)
generator = MajoranaRepresentationHamiltonianGenerator(**generator_params)

autoencoder = DiffusionAutoencoder(encoder, generator, latent_shapes=params['latent_shapes'])

noise_generator = NoiseGenerator(min_inter_site_interaction_range=2, max_inter_site_interaction_range=3, representation_mapping=RepresentationMapping.none, site_constant=True)

print(autoencoder)

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=generator_params['lr'])


save_data_list(['Epoch', 'Training loss'], loss_path, mode='w')


for epoch in range(0, params['epochs'] + 1):
    train_loss = train_diffusion_model(
        autoencoder,
        noise_generator,
        train_loader,
        optimizer,
        device,
        epoch,
    )
    save_data_list([epoch, train_loss], loss_path)
    
    # sample and save every 50 epochs
    if epoch % 50 == 0:
        save_autoencoder(encoder, generator, root_dir, epoch)
        generator.to(device)
        test_matrix_path = os.path.join(root_dir, 'tests', f'test_{epoch}')
        os.makedirs(test_matrix_path, exist_ok=True)
        num_states = 5
        autoencoder.eval()
        hamiltonian_dim = params['N'] * params['block_size']
        # context_vector = torch.tensor([0.5]) # enforcing presence of Majoranas
        samples = sample_ddpm(autoencoder, noise_generator, num_states, matrix_dim=hamiltonian_dim, device=device) #, context_vector=context_vector)

        # denormalize before plotting
        denormalization = Denormalize(mean, std)
        denormalized_samples = denormalization(samples)

        for i in range(num_states):
            matrix = denormalized_samples[i].detach().cpu().numpy()
            
            plot_matrix(matrix[0], os.path.join(test_matrix_path, f"random_hamiltonian_real_{i}.png"), vmin=-vscale, vmax=vscale)
            plot_matrix(matrix[1], os.path.join(test_matrix_path, f"random_hamiltonian_imag_{i}.png"), vmin=-vscale, vmax=vscale)

            # Plot eigvals
            eigvals_gen_plot_name = f'eigvals_spectre_{i}.png'
            eigvals_gen_plot_path = os.path.join(test_matrix_path, eigvals_gen_plot_name)
            ham = TorchHamiltonian.from_2channel_tensor(denormalized_samples[i])
            plot_eigvals_levels(ham, save_path=eigvals_gen_plot_path, ynorm=ynorm, ylim=ylim)

   
plot_convergence(loss_path, convergence_path, read_label=True)
