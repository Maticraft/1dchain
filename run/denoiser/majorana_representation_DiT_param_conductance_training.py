import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader, Subset, RandomSampler
import torch
from torchvision.transforms import Normalize

from src.data.datasets import HamiltionianDataset
from src.data.utils import calculate_mean_and_std, Denormalize
from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.conductance import plot_conductance_map
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonianParameters, QuantumDotsHamiltonian
from src.hamiltonian.utils import plot_eigvals_levels
from src.models.noise_generatiron import NoiseGenerator
from src.models.diffusion import DiffusionAutoencoder
from src.models.denoiser import train_conductance_model, test_conductance_model
from src.models.files import save_params, save_model, save_data_list
from src.models.diffusion_transformer import DiT
from src.plots import plot_convergence, plot_matrix
from src.hamiltonian.utils import plot_eigvals_levels
from src.hamiltonian.torch_hamiltonian import TorchHamiltonian



# Paths
data_path = './data/quantum_dots/3dots1level_majoranas_gap_pol_verified_with_conductance'
data_mean_std_path = f'{data_path}/mean_std.pkl'
save_dir = './conductance/quantum_dots/3dots1level_majoranas_gap_pol_verified'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_majoranas_latent_ep_{}'
eigvals_plot_name = 'eigvals_spectre_{}.png'
hamiltonian_plot_name = 'hamiltonian_{}.png'

# Reference eigvals plot params
tests_sub_dir = 'tests'
x_axis = 'mu'
x_values = np.linspace(-1.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-5., 5.)
vscale = 1/AtomicUnits.Eh

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model name
model_name = 'QDH-1lvl-no-interlevel_DiT_12_ham_flat_param_lr1e-4decreasing'

# Params
params = {
    'epochs': 500,
    'batch_size': 32,
    'lr': 1e-4,
    'max_noise_amplitude': 0.1,
}

# Architecture
dit_config = {
    'input_size': 200,
    'output_size': 12,
    'patch_size': 4,
    'hidden_size': 256,
    'depth': 6,
    'num_heads': 8,
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

tests_sub_path = os.path.join(root_dir, tests_sub_dir)     
if not os.path.isdir(tests_sub_path):
    os.makedirs(tests_sub_path)

# Try to load data statistics
try:
    with open(data_mean_std_path, 'rb') as f:
        mean, std = pickle.load(f)
except:
    data = HamiltionianDataset(data_path, label_idx=[1, 2], format='csr', threshold=0.05, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)
    data_loader = DataLoader(data, params['batch_size'])
    mean, std = calculate_mean_and_std(data_loader, device=device)
    with open(data_mean_std_path, 'wb') as f:
        pickle.dump((mean, std), f)

print('Data mean:', mean)
print('Data std:', std)

data = HamiltionianDataset(data_path, cmap='cmap0', label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])# train_data = Subset(data, range(len(data) // 2, len(data) // 2 + 1))

# sampler = RandomSampler(train_data, replacement=True, num_samples=1000)
train_loader = DataLoader(data, params['batch_size']) #, sampler=sampler)
test_loader = DataLoader(test_data, params['batch_size'])

model = DiT(**dit_config)

noise_generator = NoiseGenerator(min_inter_site_interaction_range=1, max_inter_site_interaction_range=2, representation_mapping=RepresentationMapping.none, site_constant=False)

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 200, 400, 600, 800])

save_data_list(['Epoch', 'Training loss', "Test denoising loss", "Test reference loss"], loss_path, mode='w')


for epoch in range(0, params['epochs'] + 1):
    train_loss = train_conductance_model(
        model,
        train_loader,
        optimizer,
        device,
        epoch,
    )
    test_loss, test_ref_loss = test_conductance_model(
        model,
        test_loader,
        device,
        epoch,
    )
    scheduler.step()
    save_data_list([epoch, train_loss, test_loss, test_ref_loss], loss_path)
    
    # sample and save every 10 epochs
    if epoch % 10 == 0:
        save_model(model, root_dir, epoch)
        model.eval()
        model.to(device)
        epoch_dir = os.path.join(tests_sub_path, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        amplitude_noise_dir = os.path.join(epoch_dir, 'amplitude_noise')
        os.makedirs(amplitude_noise_dir, exist_ok=True)

        full_noise_dir = os.path.join(epoch_dir, 'full_noise')
        os.makedirs(full_noise_dir, exist_ok=True)

        eigvals_test_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'reference'))
        h_torch_normalized = test_data[0][0][0].unsqueeze(0).to(device)
        h_torch_denormalized = Denormalize(mean=mean, std=std)(h_torch_normalized)[0]
        test_hamiltonian = TorchHamiltonian.from_2channel_tensor(h_torch_denormalized)
        plot_eigvals_levels(test_hamiltonian, save_path=eigvals_test_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)

        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_torch_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('ref_real'), vmin=-vscale, vmax=vscale)
        plot_matrix(h_torch_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('ref_imag'), vmin=-vscale, vmax=vscale)
        
        eigvals_dit_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'DiT'))
        eigvals_noisy_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'noisy'))

        h_torch_noisy = test_data[0][0][1].unsqueeze(0).to(device)

        noise_amplitude = torch.zeros(1, 1).to(device)
        h_denoised = model(h_torch_noisy, noise_amplitude, None)
        h_denoised = Denormalize(mean=mean, std=std)(h_denoised)[0]
        
        ham_denoised = TorchHamiltonian.from_2channel_tensor(h_denoised)
        plot_eigvals_levels(ham_denoised, save_path=eigvals_dit_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)
        
        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_denoised[0].detach().cpu().numpy(), test_matrix_path.format('denoised_real'), vmin=-vscale, vmax=vscale)
        plot_matrix(h_denoised[1].detach().cpu().numpy(), test_matrix_path.format('denoised_imag'), vmin=-vscale, vmax=vscale)
        plot_conductance_map(h_torch_noisy[0, 0].detach().cpu().numpy(), os.path.join(epoch_dir, 'conductance_map.png'), xlabel="$V$ [mV]", ylabel="$E_F$ [meV]")


plot_convergence(loss_path, convergence_path, read_label=True)
