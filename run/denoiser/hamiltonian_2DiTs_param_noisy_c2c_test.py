import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.transforms import Normalize

from src.data.datasets import HamiltonianFromParametersDataset
from src.data.utils import calculate_mean_and_std, Denormalize
from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.conductance import plot_conductance_map, generate_conductance_tensor
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.hamiltonian.hamiltonian_torch_handlers import HamiltonianConverter
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonian
from src.hamiltonian.utils import plot_eigvals_levels
from src.models.files import load_params, load_model
from src.models.diffusion_transformer import DiT
from src.models.utils import deep_update
from src.plots import plot_matrix
from src.hamiltonian.utils import plot_eigvals_levels
from src.hamiltonian.torch_hamiltonian import TorchHamiltonian

# Paths
data_path = './data/quantum_dots/3dots1level_majoranas_gap_pol_verified_with_conductance'
normalization_params_path = f'{data_path}/std_rep_normalization_params_4maps.pkl'
save_dir = './conductance/quantum_dots/3dots1level_majoranas_gap_pol_verified'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_majoranas_latent_ep_{}'
eigvals_plot_name = 'eigvals_spectre_{}.png'
hamiltonian_plot_name = 'hamiltonian_{}.png'

# Tests params
num_samples = 10
epoch = 350
tests_sub_dir = f'noisy_reconstruction_tests_ep_{epoch}'

# Plot params
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-5., 5.)
vscale = 1/AtomicUnits.Eh

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model name
model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_2DiT-c2h-12-from-params-gradually-increasing-noise_h2c-12-from-matrix_mse-loss_4maps50x50embed_norm-c2c_lr1e-4-decreasing'

# Params
params = {
    'epochs': 500,
    'batch_size': 16,
    'lr': 1e-4,
    'max_noise_amplitude': 1.,
    'random_noise': True,
}


defaults = DefaultParameters()
l = params['max_noise_amplitude']
params_noise_config = {
    'parameters': {
        "mu": (l, defaults.mu_range[1]),
        "t": (l, defaults.t_range[1]),
        "b": (l, defaults.b_range[1]),
        "d": (l, defaults.d_range[1]),
        "ph_d": (l, defaults.ph_d_range[1]),
        "l": (l, defaults.l_range[1]),
        "l_rho": (l, defaults.l_rho_range[1]),
        "l_ksi": (l, defaults.l_ksi_range[1]),
    }
}

conductance_config = {
    'cmap_list': [
    {
        'cmap2': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'b_range': (0./AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'ef_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'b_num': 50,
            'ef_num': 50,
            'with_embedding': False
        },
    },
    {
        'cmap2': {
            'i':0,
            'j':1,
            'gamma': 0.1,
            'b_range': (0./AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'ef_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'b_num': 50,
            'ef_num': 50,
            'with_embedding': False
        },
    },
        {
        'cmap2': {
            'i':1,
            'j':0,
            'gamma': 0.1,
            'b_range': (0./AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'ef_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'b_num': 50,
            'ef_num': 50,
            'with_embedding': False
        },
    },
    {
        'cmap2': {
            'i':1,
            'j':1,
            'gamma': 0.1,
            'b_range': (0./AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'ef_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'b_num': 50,
            'ef_num': 50,
            'with_embedding': True
        },
    },
    ],
}

# Set the root dir
root_dir = os.path.join(save_dir, f'DiT', model_name)
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)


tests_sub_path = os.path.join(root_dir, tests_sub_dir)     
if not os.path.isdir(tests_sub_path):
    os.makedirs(tests_sub_path)

# Try to load data statistics
try:
    with open(normalization_params_path, 'rb') as f:
        normalization_params = pickle.load(f)
except:
    data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, conductance_config=conductance_config, random_noise=params['random_noise'])
    data_loader = DataLoader(data, params['batch_size'])
    normalization_params = calculate_mean_and_std(data_loader, device=device)
    with open(normalization_params_path, 'wb') as f:
        pickle.dump(normalization_params, f)

print('Normalization params:', normalization_params)

# data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down, conductance_config=conductance_config, noisy_conductance=True, target_condcuctance=True)
data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_params=normalization_params, conductance_config=conductance_config, random_noise=params['random_noise'])

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])# train_data = Subset(data, range(len(data) // 2, len(data) // 2 + 1))

dit_c2h_config = load_params(os.path.join(root_dir, 'dit_c2h_config.json'))
dit_h2c_config = load_params(os.path.join(root_dir, 'dit_h2c_config.json'))

model_c2h = load_model(DiT, dit_c2h_config, root_dir, epoch=epoch, suffix='_c2h')
model_h2c = load_model(DiT, dit_h2c_config, root_dir, epoch=epoch, suffix='_h2c')

model_c2h.eval()
model_c2h.to(device)

model_h2c.eval()
model_h2c.to(device)

hamiltonian_converter = HamiltonianConverter(
    dit_h2c_config['input_hamiltonian_params'],
    min_inter_site_interaction_range=dit_h2c_config['min_inter_site_interaction_range'],
    max_inter_site_interaction_range=dit_h2c_config['max_inter_site_interaction_range'],
)

h_mean, h_std = normalization_params[0]
h_denormalize = Denormalize(mean=h_mean, std=h_std)

cmap_mean, cmap_std = normalization_params[1]
cmap_normalize = Normalize(mean=cmap_mean, std=cmap_std)
cmap_denormalize = Denormalize(mean=cmap_mean, std=cmap_std)

noisy_cmap_mean, noisy_cmap_std = normalization_params[3]
noisy_cmap_denormalize = Denormalize(mean=noisy_cmap_mean, std=noisy_cmap_std)

for sample_idx in range(num_samples):    
    sample_dir = os.path.join(tests_sub_path, f'sample_{sample_idx}')
    sample_data = test_data[sample_idx]

    os.makedirs(sample_dir, exist_ok=True)

    eigvals_test_path = os.path.join(sample_dir, eigvals_plot_name.format(f'reference'))
    h_torch_normalized = sample_data[0][0].unsqueeze(0).to(device)
    h_torch_denormalized = h_denormalize(h_torch_normalized)[0]
    test_hamiltonian = TorchHamiltonian.from_2channel_tensor(h_torch_denormalized)
    plot_eigvals_levels(test_hamiltonian, save_path=eigvals_test_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)

    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_torch_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('ref_real'), vmin=-vscale, vmax=vscale)
    plot_matrix(h_torch_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('ref_imag'), vmin=-vscale, vmax=vscale)
    
    eigvals_noisy_path = os.path.join(sample_dir, eigvals_plot_name.format(f'noisy'))
    h_perturbed = sample_data[0][2].unsqueeze(0).to(device)
    h_noisy_denormalized = h_denormalize(h_perturbed)[0]
    test_noisy_hamiltonian = TorchHamiltonian.from_2channel_tensor(h_noisy_denormalized)
    plot_eigvals_levels(test_noisy_hamiltonian, save_path=eigvals_noisy_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)

    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_noisy_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('noisy_real'), vmin=-vscale, vmax=vscale)
    plot_matrix(h_noisy_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('noisy_imag'), vmin=-vscale, vmax=vscale)
    

    mock_noise_amplitude = torch.zeros(1, 1).to(device)
    real_noise_amplitude = torch.tensor(sample_data[1][1]).to(device).view(1, 1)

    h_perturbed_params_map = hamiltonian_converter.from_matrix_to_params(h_perturbed)

    h_noisy_conductance = sample_data[0][3].unsqueeze(0).to(device)
    params_map = model_c2h(h_noisy_conductance, real_noise_amplitude, None, matrix_output=False)
    # improved_map = deep_update(h_perturbed_params_map, params_map, detach=False)
    # improved_map = weighted_update(h_perturbed_params_map, params_map, alpha=0.5)
    improved_map = params_map

    h_predicted = hamiltonian_converter.from_params_to_matrix(improved_map)
    h_predicted_denorm = h_denormalize(h_predicted)[0]
    
    eigvals_dit_path = os.path.join(sample_dir, eigvals_plot_name.format(f'DiT_rec'))

    ham_rec = TorchHamiltonian.from_2channel_tensor(h_predicted_denorm)
    plot_eigvals_levels(ham_rec, save_path=eigvals_dit_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)
    
    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_predicted_denorm[0].detach().cpu().numpy(), test_matrix_path.format('improved_real'), vmin=-vscale, vmax=vscale)
    plot_matrix(h_predicted_denorm[1].detach().cpu().numpy(), test_matrix_path.format('improved_imag'), vmin=-vscale, vmax=vscale)
    
    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        noisy_conductance_path = os.path.join(sample_dir, 'noisy_conductance')
        os.makedirs(noisy_conductance_path, exist_ok=True)
        x_tick_range = cmap_config['cmap2']['b_range']
        y_tick_range = cmap_config['cmap2']['ef_range']
        i_val = cmap_config['cmap2']['i']
        j_val = cmap_config['cmap2']['j']
        h_noisy_conductance = noisy_cmap_denormalize(h_noisy_conductance)
        plot_conductance_map(
            h_noisy_conductance[0, i].detach().cpu().numpy(),
            os.path.join(noisy_conductance_path, f'map_i{i_val}_j{j_val}.png'),
            xtick_range=x_tick_range,
            ytick_range=y_tick_range,
            xlabel="$B$ [mV]",
            ylabel="$E_F$ (meV)"
        )

    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        ref_conductance_path = os.path.join(sample_dir, 'ref_conductance')
        os.makedirs(ref_conductance_path, exist_ok=True)
        x_tick_range = cmap_config['cmap2']['b_range']
        y_tick_range = cmap_config['cmap2']['ef_range']
        i_val = cmap_config['cmap2']['i']
        j_val = cmap_config['cmap2']['j']
        ref_cmap = sample_data[0][1].to(device)
        ref_cmap = cmap_denormalize(ref_cmap)
        plot_conductance_map(
            ref_cmap[i].detach().cpu().numpy(),
            os.path.join(ref_conductance_path, f'map_i{i_val}_j{j_val}.png'),
            xtick_range=x_tick_range,
            ytick_range=y_tick_range,
            xlabel="$B$ [mV]",
            ylabel="$E_F$ (meV)"
        )

    mapped_h_predicted = torch.complex(h_predicted_denorm[0], h_predicted_denorm[1])
    predicted_cmap = generate_conductance_tensor(mapped_h_predicted, conductance_config)
    
    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        predicted_conductance_path = os.path.join(sample_dir, 'improved_conductance')
        os.makedirs(predicted_conductance_path, exist_ok=True)
        x_tick_range = cmap_config['cmap2']['b_range']
        y_tick_range = cmap_config['cmap2']['ef_range']
        i_val = cmap_config['cmap2']['i']
        j_val = cmap_config['cmap2']['j']
        plot_conductance_map(
            predicted_cmap[i].detach().cpu().numpy(),
            os.path.join(predicted_conductance_path, f'map_i{i_val}_j{j_val}.png'),
            xtick_range=x_tick_range,
            ytick_range=y_tick_range,
            xlabel="$B$ [mV]",
            ylabel="$E_F$ (meV)"
        )


    # H2C evaluation
    model_h2c.eval()
    model_h2c.to(device)

    cmap = model_h2c(improved_map, mock_noise_amplitude, None, matrix_input=False)
    cmap = torch.cat((cmap, torch.zeros((cmap.shape[0], 2, *cmap.shape[2:]), device=device)), dim=1)
    cmap_denorm = cmap_denormalize(cmap)[0]

    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        cmap_h2c_conductance_path = os.path.join(sample_dir, 'h2c_from_c2h_conductance')
        os.makedirs(cmap_h2c_conductance_path, exist_ok=True)
        x_tick_range = cmap_config['cmap2']['b_range']
        y_tick_range = cmap_config['cmap2']['ef_range']
        i_val = cmap_config['cmap2']['i']
        j_val = cmap_config['cmap2']['j']
        plot_conductance_map(
            cmap_denorm[i].detach().cpu().numpy(),
            os.path.join(cmap_h2c_conductance_path, f'map_i{i_val}_j{j_val}.png'),
            xtick_range=x_tick_range,
            ytick_range=y_tick_range,
            xlabel="$B$ [mV]",
            ylabel="$E_F$ (meV)"
        )


    cmap = model_h2c(h_torch_normalized, mock_noise_amplitude, None)
    cmap = torch.cat((cmap, torch.zeros((cmap.shape[0], 2, *cmap.shape[2:]), device=device)), dim=1)
    cmap_denorm = cmap_denormalize(cmap)[0]

    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        cmap_h2c_conductance_path = os.path.join(sample_dir, 'h2c_from_ref_conductance')
        os.makedirs(cmap_h2c_conductance_path, exist_ok=True)
        x_tick_range = cmap_config['cmap2']['b_range']
        y_tick_range = cmap_config['cmap2']['ef_range']
        i_val = cmap_config['cmap2']['i']
        j_val = cmap_config['cmap2']['j']
        plot_conductance_map(
            cmap_denorm[i].detach().cpu().numpy(),
            os.path.join(cmap_h2c_conductance_path, f'map_i{i_val}_j{j_val}.png'),
            xtick_range=x_tick_range,
            ytick_range=y_tick_range,
            xlabel="$B$ [mV]",
            ylabel="$E_F$ (meV)"
        )