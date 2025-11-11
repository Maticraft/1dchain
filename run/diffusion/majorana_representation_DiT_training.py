import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader, Subset, RandomSampler
import torch

from src.data.datasets import HamiltionianDataset
from src.data.utils import calculate_mean_and_std, Denormalize
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.hamiltonian.units import AtomicUnits
from src.models.noise_generatiron import NoiseGenerator
from src.models.diffusion import DiffusionAutoencoder, train_diffusion_model, sample_ddpm
from src.models.files import save_params, save_model, save_data_list
from src.models.diffusion_transformer import DiT
from src.plots import plot_convergence, plot_matrix, plot_generator_eigvals
from src.hamiltonian.utils import plot_eigvals_levels
from src.hamiltonian.torch_hamiltonian import TorchHamiltonian

# Paths
data_path = './data/quantum_dots/3dots1level_majoranas_gap_pol_verified'
data_mean_std_path = f'{data_path}/mean_std.pkl'
save_dir = './diffusion/quantum_dots/3dots1level_majoranas_gap_pol_verified'
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
model_name = 'QDH-1lvl-no-interlevel_DiT_28_ham_flat_param_embed_pol_ctx_time_period500_lr1e-4decreasing'

# Params
params = {
    'epochs': 500,
    'batch_size': 64,
    'lr': 1e-4,
}

# Architecture
dit_config = {
    'input_size': 12,
    'patch_size': 4,
    'hidden_size': 1024,
    'depth': 16,
    'num_heads': 16,
    'mlp_ratio': 4.0,
    'class_dropout_prob': 0.1,
    'num_classes': 2,
    'learn_sigma': False,
    'min_inter_site_interaction_range': 1,
    'max_inter_site_interaction_range': 2,
}


# Set the root dir
root_dir = os.path.join(save_dir, f'DiT', model_name)
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

save_params(dit_config, os.path.join(root_dir, 'dit_config.json'))

loss_path = os.path.join(root_dir, loss_file)
convergence_path = os.path.join(root_dir, convergence_file)

eigvals_sub_path = os.path.join(root_dir, eigvals_sub_dir)     
if not os.path.isdir(eigvals_sub_path):
    os.makedirs(eigvals_sub_path)


# Try to load data statistics
try:
    with open(data_mean_std_path, 'rb') as f:
        mean, std = pickle.load(f)
except:
    data = HamiltionianDataset(data_path, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
    data_loader = DataLoader(data, params['batch_size'])
    mean, std = calculate_mean_and_std(data_loader, device=device)
    with open(data_mean_std_path, 'wb') as f:
        pickle.dump((mean, std), f)

print('Data mean:', mean)
print('Data std:', std)

data = HamiltionianDataset(data_path, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
# train_data = Subset(data, range(len(data) // 2, len(data) // 2 + 1))

# sampler = RandomSampler(train_data, replacement=True, num_samples=1000)
train_loader = DataLoader(data, params['batch_size']) #, sampler=sampler)

model = DiT(**dit_config)

noise_generator = NoiseGenerator(min_inter_site_interaction_range=1, max_inter_site_interaction_range=2, representation_mapping=RepresentationMapping.none, site_constant=False)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200, 400, 600, 800])

save_data_list(['Epoch', 'Training loss'], loss_path, mode='w')


for epoch in range(0, params['epochs'] + 1):
    train_loss = train_diffusion_model(
        model,
        noise_generator,
        train_loader,
        optimizer,
        device,
        epoch,
    )
    scheduler.step()
    save_data_list([epoch, train_loss], loss_path)
    
    # sample and save every 10 epochs
    if epoch % 10 == 0:
        save_model(model, root_dir, epoch)
        model.to(device)
        test_matrix_path = os.path.join(root_dir, 'tests', f'test_{epoch}')
        os.makedirs(test_matrix_path, exist_ok=True)
        num_states = 5
        model.eval()
        hamiltonian_dim = dit_config['input_size']
        context_vector = torch.tensor([1.0, 1.0]) # enforcing presence of Majoranas
        samples = sample_ddpm(model, noise_generator, num_states, matrix_dim=hamiltonian_dim, device=device, context_vector=context_vector)

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
