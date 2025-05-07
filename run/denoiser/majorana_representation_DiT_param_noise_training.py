import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader, Subset, RandomSampler
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
from src.models.diffusion import DiffusionAutoencoder
from src.models.denoiser import train_denoising_param_model, test_denoising_param_model
from src.models.files import save_params, save_model, save_data_list
from src.models.diffusion_transformer import DiT
from src.plots import plot_convergence, plot_matrix
from src.hamiltonian.utils import plot_eigvals_levels
from src.torch_utils import TorchHamiltonian

# Paths
data_path = './data/quantum_dots/3dots1level_majoranas_separated'
data_mean_std_path = f'{data_path}/mean_std.pkl'
save_dir = './param_denoiser/quantum_dots/3dots1level_majoranas_separated'
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
model_name = 'QDH-1lvl-no-interlevel_DiT_28_ham_flat_param_noise_strength_random_gauss_max1_lr1e-4decreasing'

# TODO: try increasing probability of small noise

# Params
params = {
    'epochs': 500,
    'batch_size': 64,
    'lr': 1e-4,
    'max_noise_amplitude': 1.,
    'random_noise': True
}

# Architecture
dit_config = {
    'input_size': 12,
    'output_size': 12,
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
    'input_embedder': 'hamiltonian',
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

# Try to load data statistics
try:
    with open(data_mean_std_path, 'rb') as f:
        mean, std = pickle.load(f)
except:
    data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05,representation_mapping=RepresentationMapping.majorana_plus_minus_up_down, random_noise=params['random_noise'])
    data_loader = DataLoader(data, params['batch_size'])
    mean, std = calculate_mean_and_std(data_loader, device=device)
    with open(data_mean_std_path, 'wb') as f:
        pickle.dump((mean, std), f)

print('Data mean:', mean)
print('Data std:', std)

data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down, random_noise=params['random_noise'])

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
    train_loss = train_denoising_param_model(
        model,
        train_loader,
        optimizer,
        device,
        epoch,
    )
    test_loss, test_ref_loss = test_denoising_param_model(
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
        
        # ------------------------------------------------------------------------------
        # Visualize the results
        # ------------------------------------------------------------------------------
        
        model.eval()
        model.to(device)
        epoch_dir = os.path.join(tests_sub_path, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        amplitude_noise_dir = os.path.join(epoch_dir, 'amplitude_noise')
        os.makedirs(amplitude_noise_dir, exist_ok=True)

        full_noise_dir = os.path.join(epoch_dir, 'full_noise')
        os.makedirs(full_noise_dir, exist_ok=True)

        eigvals_test_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'reference'))
        
        sample = test_data[0]
        
        h_torch_normalized = sample[0][0].unsqueeze(0).to(device)
        h_torch_denormalized = Denormalize(mean=mean, std=std)(h_torch_normalized)[0]
        test_hamiltonian = TorchHamiltonian.from_2channel_tensor(h_torch_denormalized)
        plot_eigvals_levels(test_hamiltonian, save_path=eigvals_test_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)
        
        polarization_sub_path = os.path.join(epoch_dir, 'polarization')
        plot_majorana_polarization(test_hamiltonian, polarization_sub_path, threshold = MZM_THRESHOLD, string_num=1, polaxis='x', representation=RepresentationMapping.majorana_plus_minus_up_down)

        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
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
            os.path.join(epoch_dir, 'ref_conductance_map.png'),
            xtick_range=x_tick_range,
            ytick_range=y_tick_range,
            xlabel="$B$ [mV]",
            ylabel="$E_F$ [meV]"
        )
        
        for dir_path in [amplitude_noise_dir, full_noise_dir]:
            eigvals_dit_path = os.path.join(dir_path, eigvals_plot_name.format(f'DiT'))
            eigvals_noisy_path = os.path.join(dir_path, eigvals_plot_name.format(f'noisy'))

            if 'full_noise' in dir_path:
                default_params = DefaultParameters(mu_max=1., t_max=1., b_max=1, d_max=1, lambda_max=1)
                hamiltonian_params = QuantumDotsHamiltonianParameters(no_dots=3, no_levels=1, default_parameters=default_params)
                random_hamiltonian_params = hamiltonian_params.set_random_parameters_free()
                hamiltonian = QuantumDotsHamiltonian(hamiltonian_params)
                h_torch_noisy = hamiltonian.get_hamiltonian_tensor(representation_mapping=RepresentationMapping.majorana_plus_minus_up_down).unsqueeze(0).to(device)
                h_torch_noisy = Normalize(mean, std)(h_torch_noisy)
                noise_amplitude = torch.ones(1, 1).to(device)
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


plot_convergence(loss_path, convergence_path, read_label=True)
