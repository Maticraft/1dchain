import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch
from torchvision.transforms import Normalize

from src.data.datasets import HamiltonianFromParametersDataset
from src.data.utils import calculate_mean_and_std, Denormalize
from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.conductance import plot_conductance_map, torch_conductance_map0, torch_conductance_map2
from src.hamiltonian.hamiltonian import RepresentationMapping, transform_majorana_plus_minus_up_down_representation_to_default
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonianParameters, QuantumDotsHamiltonian, MZM_THRESHOLD
from src.hamiltonian.utils import plot_eigvals_levels, plot_majorana_polarization
from src.models.noise_generatiron import NoiseGenerator
from src.models.denoiser import train_denoising_param_model, test_denoising_param_model, diffusion_like_sample
from src.models.files import load_model, save_params, save_model, save_data_list, load_params
from src.models.diffusion_transformer import DiT
from src.plots import plot_convergence, plot_matrix
from src.hamiltonian.utils import plot_eigvals_levels
from src.torch_utils import TorchHamiltonian

# Paths
data_path = './data/quantum_dots/3dots1level_majoranas_separated'
data_mean_std_path = f'{data_path}/mean_std.pkl'
save_dir = './param_denoiser/quantum_dots/3dots1level_majoranas_separated'
loss_file = 'loss.txt'
distribution_dir_name = 'tests_majoranas_latent_ep_{}'
eigvals_plot_name = 'eigvals_spectre_{}.png'
hamiltonian_plot_name = 'hamiltonian_{}.png'

# Reference eigvals plot params
tests_sub_dir = 'diffusion_generation_tests'
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
model_name = 'QDH-1lvl-no-interlevel_DiT_28_ham_flat_param_noise_strength_random_gauss_max1_lr1e-4decreasing'

# Params
params = {
    'epochs': 500,
    'batch_size': 64,
    'lr': 1e-4,
    'max_noise_amplitude': 1.,
    'random_noise': True
}


# Set the root dir
root_dir = os.path.join(save_dir, f'DiT', model_name)
tests_sub_path = os.path.join(root_dir, tests_sub_dir)     
if not os.path.isdir(tests_sub_path):
    os.makedirs(tests_sub_path)


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
    'cmap2': {
        'i':0,
        'j':0,
        'gamma': 0.1,
        'b_range': (0./AtomicUnits.Eh, 2./AtomicUnits.Eh),
        'ef_range': (-2./AtomicUnits.Eh, 2./AtomicUnits.Eh),
        'b_num': 200,
        'ef_num': 200
    }
}
x_tick_range = conductance_config['cmap2']['b_range']
y_tick_range = conductance_config['cmap2']['ef_range']

with open(data_mean_std_path, 'rb') as f:
    mean, std = pickle.load(f)

print('Data mean:', mean)
print('Data std:', std)

data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down, random_noise=params['random_noise'])

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])# train_data = Subset(data, range(len(data) // 2, len(data) // 2 + 1))


dit_config = load_params(os.path.join(root_dir, 'dit_config.json'))
model = load_model(DiT, dit_config, root_dir, epoch=500) 

print(model)

model.eval()
model.to(device)


for i in range(n_samples_to_plot):        
    # ------------------------------------------------------------------------------
    # Visualize the results
    # ------------------------------------------------------------------------------
    
    model.eval()
    model.to(device)
    sample_dir = os.path.join(tests_sub_path, f'sample_{i}')
    os.makedirs(sample_dir, exist_ok=True)

    amplitude_noise_dir = os.path.join(sample_dir, 'amplitude_noise')
    os.makedirs(amplitude_noise_dir, exist_ok=True)

    full_noise_dir = os.path.join(sample_dir, 'full_noise')
    os.makedirs(full_noise_dir, exist_ok=True)

    eigvals_test_path = os.path.join(sample_dir, eigvals_plot_name.format(f'reference'))
    
    sample = test_data[i]
    
    h_torch_normalized = sample[0][0].unsqueeze(0).to(device)
    h_torch_denormalized = Denormalize(mean=mean, std=std)(h_torch_normalized)[0]
    test_hamiltonian = TorchHamiltonian.from_2channel_tensor(h_torch_denormalized)
    plot_eigvals_levels(test_hamiltonian, save_path=eigvals_test_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)
    
    polarization_sub_path = os.path.join(sample_dir, 'polarization')
    plot_majorana_polarization(test_hamiltonian, polarization_sub_path, threshold = MZM_THRESHOLD, string_num=1, polaxis='x', representation=RepresentationMapping.majorana_plus_minus_up_down)

    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_torch_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('ref_real'), vmin=-vscale, vmax=vscale)
    plot_matrix(h_torch_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('ref_imag'), vmin=-vscale, vmax=vscale)

    # Reference conductance map
    mapped_h_ref = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(h_torch_denormalized[0], h_torch_denormalized[1]))
    if "cmap0" in conductance_config:
        ref_cmap = torch_conductance_map0(mapped_h_ref.unsqueeze(0), **conductance_config['cmap0'])
    if "cmap2" in conductance_config:
        ref_cmap = torch_conductance_map2(mapped_h_ref.unsqueeze(0), **conductance_config['cmap2'])
    plot_conductance_map(
        ref_cmap[0].detach().cpu().numpy(),
        os.path.join(sample_dir, 'ref_conductance_map.png'),
        xtick_range=x_tick_range,
        ytick_range=y_tick_range,
        xlabel="$B$ [mV]",
        ylabel="$E_F$ [meV]"
    )
    
    for dir_path in [amplitude_noise_dir, full_noise_dir]:
        eigvals_dit_path = os.path.join(dir_path, eigvals_plot_name.format(f'DiT'))
        eigvals_noisy_path = os.path.join(dir_path, eigvals_plot_name.format(f'noisy'))

        if 'full_noise' in dir_path:
            def noisy_sample():
                default_params = DefaultParameters(mu_max=1., t_max=1., b_max=1, d_max=1, lambda_max=1)
                hamiltonian_params = QuantumDotsHamiltonianParameters(no_dots=3, no_levels=1, default_parameters=default_params)
                hamiltonian_params.set_random_parameters_free()
                hamiltonian = QuantumDotsHamiltonian(hamiltonian_params)
                h_torch_noisy = hamiltonian.get_hamiltonian_tensor(representation_mapping=RepresentationMapping.majorana_plus_minus_up_down).unsqueeze(0).to(device)
                h_torch_noisy = Normalize(mean, std)(h_torch_noisy)
                return h_torch_noisy
            
            h_torch_noisy = noisy_sample()
            h_denoised = diffusion_like_sample(model, noisy_sample, device=device, first_noisy_sample=h_torch_noisy)
            # noise_amplitude = torch.ones(1, 1).to(device)
        else:
            h_torch_noisy = sample[0][2].unsqueeze(0).to(device)
            noise_amplitude = torch.tensor(sample[1][1]).view(1, 1).to(device)
            h_denoised = model(h_torch_noisy, noise_amplitude, None)

        h_denoised = Denormalize(mean=mean, std=std)(h_denoised)[0]
        h_noisy_denormalized = Denormalize(mean=mean, std=std)(h_torch_noisy)[0]
        
        # Noisy eigvals plot
        ham_noised = TorchHamiltonian.from_2channel_tensor(h_noisy_denormalized)
        plot_eigvals_levels(ham_noised, save_path=eigvals_noisy_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)
        
        # Denoised eigvals plot
        ham_denoised = TorchHamiltonian.from_2channel_tensor(h_denoised)
        plot_eigvals_levels(ham_denoised, save_path=eigvals_dit_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)
        
        test_matrix_path = os.path.join(dir_path, hamiltonian_plot_name)
        # Noisy hamiltonian  
        plot_matrix(h_noisy_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('noisy_real'), vmin=-vscale, vmax=vscale)
        plot_matrix(h_noisy_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('noisy_imag'), vmin=-vscale, vmax=vscale)

        # Denoised hamiltonian
        plot_matrix(h_denoised[0].detach().cpu().numpy(), test_matrix_path.format('denoised_real'), vmin=-vscale, vmax=vscale)
        plot_matrix(h_denoised[1].detach().cpu().numpy(), test_matrix_path.format('denoised_imag'), vmin=-vscale, vmax=vscale)

        # Noisy polarization
        noisy_polarization_sub_path = os.path.join(dir_path, 'noisy_polarization')
        plot_majorana_polarization(ham_noised, noisy_polarization_sub_path, threshold = MZM_THRESHOLD, string_num=1, polaxis='x', representation=RepresentationMapping.majorana_plus_minus_up_down)

        # Denoised polarization
        denoised_polarization_sub_path = os.path.join(dir_path, 'denoised_polarization')
        plot_majorana_polarization(ham_denoised, denoised_polarization_sub_path, threshold = MZM_THRESHOLD, string_num=1, polaxis='x', representation=RepresentationMapping.majorana_plus_minus_up_down)

        # Noisy conductance map
        mapped_h_noisy = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(h_noisy_denormalized[0], h_noisy_denormalized[1]))
        if "cmap0" in conductance_config:
            noisy_cmap = torch_conductance_map0(mapped_h_noisy.unsqueeze(0), **conductance_config['cmap0'])
        if "cmap2" in conductance_config:
            noisy_cmap = torch_conductance_map2(mapped_h_noisy.unsqueeze(0), **conductance_config['cmap2'])
        plot_conductance_map(
            noisy_cmap[0].detach().cpu().numpy(),
            os.path.join(dir_path, 'noisy_conductance_map.png'),
            xtick_range=x_tick_range,
            ytick_range=y_tick_range,
            xlabel="$B$ [mV]",
            ylabel="$E_F$ [meV]"
        )

        # Denoised conductance map
        mapped_h_denoised = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(h_denoised[0], h_denoised[1]))
        if "cmap0" in conductance_config:
            denoised_cmap = torch_conductance_map0(mapped_h_denoised.unsqueeze(0), **conductance_config['cmap0'])
        if "cmap2" in conductance_config:
            denoised_cmap = torch_conductance_map2(mapped_h_denoised.unsqueeze(0), **conductance_config['cmap2'])
        plot_conductance_map(
            denoised_cmap[0].detach().cpu().numpy(),
            os.path.join(dir_path, 'denoised_conductance_map.png'),
            xtick_range=x_tick_range,
            ytick_range=y_tick_range,
            xlabel="$B$ [mV]",
            ylabel="$E_F$ [meV]"
        )
