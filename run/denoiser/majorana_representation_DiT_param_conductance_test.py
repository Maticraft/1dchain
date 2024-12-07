import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader, Subset, RandomSampler
import torch
from torchvision.transforms import Normalize

from src.data.datasets import HamiltionianDataset
from src.data.utils import calculate_mean_and_std, Denormalize
from src.hamiltonian.conductance import plot_conductance_map, torch_conductance_map0
from src.hamiltonian.hamiltonian import RepresentationMapping, transform_majorana_plus_minus_up_down_representation_to_default
from src.hamiltonian.quantum_dots_chain import AtomicUnits, DefaultParameters, QuantumDotsHamiltonianParameters, QuantumDotsHamiltonian
from src.hamiltonian.utils import plot_eigvals_levels
from src.models.noise_generatiron import NoiseGenerator
from src.models.diffusion import DiffusionAutoencoder
from src.models.denoiser import train_denoising_param_model, test_denoising_param_model
from src.models.files import save_params, load_model, save_data_list
from src.models.diffusion_transformer import DiT
from src.plots import plot_convergence, plot_matrix
from src.hamiltonian.utils import plot_eigvals_levels
from src.torch_utils import TorchHamiltonian

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
tests_sub_dir = 'conductance_tests'
x_axis = 'mu'
x_values = np.linspace(-1.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-5., 5.)
vscale = 1/AtomicUnits.Eh
n_samples_to_plot = 10

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
tests_sub_path = os.path.join(root_dir, tests_sub_dir)     
if not os.path.isdir(tests_sub_path):
    os.makedirs(tests_sub_path)

with open(data_mean_std_path, 'rb') as f:
    mean, std = pickle.load(f)

print('Data mean:', mean)
print('Data std:', std)

data = HamiltionianDataset(data_path, cmap='cmap0', label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down)

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])# train_data = Subset(data, range(len(data) // 2, len(data) // 2 + 1))


model = load_model(DiT, dit_config, root_dir, epoch=500) 
print(model)

model.eval()
model.to(device)

for i in range(n_samples_to_plot):
    sample_dir = os.path.join(tests_sub_path, 'sample_{}'.format(i))
    os.makedirs(sample_dir, exist_ok=True)

    eigvals_test_path = os.path.join(sample_dir, eigvals_plot_name.format(f'reference'))
    h_torch_normalized = test_data[i][0][0].unsqueeze(0).to(device)
    h_torch_denormalized = Denormalize(mean=mean, std=std)(h_torch_normalized)[0]
    test_hamiltonian = TorchHamiltonian.from_2channel_tensor(h_torch_denormalized)
    plot_eigvals_levels(test_hamiltonian, save_path=eigvals_test_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)

    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_torch_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('ref_real'), vmin=-vscale, vmax=vscale)
    plot_matrix(h_torch_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('ref_imag'), vmin=-vscale, vmax=vscale)

    eigvals_dit_path = os.path.join(sample_dir, eigvals_plot_name.format(f'DiT'))
    eigvals_noisy_path = os.path.join(sample_dir, eigvals_plot_name.format(f'noisy'))

    h_conductance = test_data[i][0][1].unsqueeze(0).to(device)

    noise_amplitude = torch.zeros(1, 1).to(device)
    h_predicted = model(h_conductance, noise_amplitude, None)
    h_predicted = Denormalize(mean=mean, std=std)(h_predicted)[0]

    ham_predicted = TorchHamiltonian.from_2channel_tensor(h_predicted)
    plot_eigvals_levels(ham_predicted, save_path=eigvals_dit_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)

    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_predicted[0].detach().cpu().numpy(), test_matrix_path.format('denoised_real'), vmin=-vscale, vmax=vscale)
    plot_matrix(h_predicted[1].detach().cpu().numpy(), test_matrix_path.format('denoised_imag'), vmin=-vscale, vmax=vscale)
    plot_conductance_map(h_conductance[0, 0].detach().cpu().numpy(), os.path.join(sample_dir, 'conductance_map.png'), xlabel="$V$ [mV]", ylabel="$E_F$ [meV]")


    mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(h_torch_denormalized[0], h_torch_denormalized[1]))
    predicted_cmap = torch_conductance_map0(
        mapped_h_predicted.unsqueeze(0),
        i=0,
        j=0,
        ef_range=(-2./AtomicUnits.Eh, 2./AtomicUnits.Eh),
        mu_range=(-2./AtomicUnits.Eh, 2./AtomicUnits.Eh),
        ef_num=200,
        mu_num=200,
        n_levels=1,
        gamma=0.1
    )
    plot_conductance_map(predicted_cmap[0].detach().cpu().numpy(), os.path.join(sample_dir, 'ref_conductance_map.png'), xlabel="$V$ [mV]", ylabel="$E_F$ [meV]")



    mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(h_predicted[0], h_predicted[1]))
    predicted_cmap = torch_conductance_map0(
        mapped_h_predicted.unsqueeze(0),
        i=0,
        j=0,
        ef_range=(-2./AtomicUnits.Eh, 2./AtomicUnits.Eh),
        mu_range=(-2./AtomicUnits.Eh, 2./AtomicUnits.Eh),
        ef_num=200,
        mu_num=200,
        n_levels=1,
        gamma=0.1
    )
    plot_conductance_map(predicted_cmap[0].detach().cpu().numpy(), os.path.join(sample_dir, 'predicted_conductance_map.png'), xlabel="$V$ [mV]", ylabel="$E_F$ [meV]")
