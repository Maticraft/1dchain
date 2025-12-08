import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader, Subset, RandomSampler
import torch
from torchvision.transforms import Normalize

from src.data.datasets import HamiltonianFromParametersDataset
from src.data.utils import calculate_mean_and_std, Denormalize
from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.conductance import plot_conductance_map, torch_conductance_map0, torch_conductance_map2, generate_conductance_tensor
from src.hamiltonian.hamiltonian import RepresentationMapping, transform_majorana_plus_minus_up_down_representation_to_default
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonianParameters, QuantumDotsHamiltonian
from src.hamiltonian.utils import plot_eigvals_levels
from src.models.noise_generatiron import NoiseGenerator
from src.models.diffusion import DiffusionAutoencoder
from src.models.denoiser import train_denoising_conductance_3models, test_denoising_conductance_3models
from src.models.files import save_params, save_model, save_data_list, load_params, load_model
from src.models.diffusion_transformer import DiT
from src.plots import plot_convergence, plot_matrix
from src.hamiltonian.utils import plot_eigvals_levels, plot_eigvals
from src.hamiltonian.torch_hamiltonian import TorchHamiltonian

# Paths
data_path = './data/quantum_dots/3dots1level_majoranas_gap_pol_verified_with_conductance'
normalization_params_path = f'{data_path}/normalization_params_4maps.pkl'
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
model_name = 'Pretrained-QDH-1lvl-no-interlevel_h2h_improvement-mask-fixed-pseudo-full-h_mse_loss_real_12_patch1_4maps50x50embed_norm-c2c_lr1e-4-equalized_decreasing'

# Pretrained model
pretrained_model_name = 'QDH-1lvl-no-interlevel_2DiTs_real_12_patch1_4maps50x50embed_norm_triple-c2c_lr1e-4decreasing'
pretrained_model_path = f'./conductance/quantum_dots/3dots1level_majoranas_gap_pol_verified/DiT/{pretrained_model_name}'
pretrained_model_epoch = 350

# Params
params = {
    'epochs': 500,
    'batch_size': 32,
    'pretrained_lr': 1e-4,
    'lr': 1e-4,
    'max_noise_amplitude': 1.,
    'model_prediction': 'hamiltonian_improvement_mask',
    'random_noise': True,
    'h2h_start_epoch': 0,
    'train_c2h': True,
    'train_h2c': True,
}


# Architecture
dit_c2h_config = {
    'input_size': 50,
    'input_channels': 6,
    'output_size': 12,
    'output_channels': 1,
    'patch_size': 1,
    'hidden_size': 256,
    'depth': 12,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'class_dropout_prob': 0.1,
    'num_classes': 1,
    'learn_sigma': False,
    'min_inter_site_interaction_range': 1,
    'max_inter_site_interaction_range': 2,
    'input_embedder': 'patch',
    'output_unpatcher': 'hamiltonian',
    # 'on_site_block_names': ['iy1'],
    # 'inter_site_block_names': ['1z 1x iy1 iyx'],
}

dit_h2c_config = {
    'input_size': 12,
    'input_channels': 1,
    'output_size': 50,
    'output_channels': 4,
    'patch_size': 1,
    'hidden_size': 256,
    'depth': 12,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'class_dropout_prob': 0.1,
    'num_classes': 1,
    'learn_sigma': False,
    'min_inter_site_interaction_range': 1,
    'max_inter_site_interaction_range': 2,
    'input_embedder': 'hamiltonian',
    'output_unpatcher': 'patch',
    # 'on_site_block_names': ['iy1'],
    # 'inter_site_block_names': ['1z 1x iy1 iyx'],
}

dit_h2h_config = {
    'input_size': 12,
    'input_channels': 1,
    'output_size': 12,
    'output_channels': 1,
    'patch_size': 4,
    'hidden_size': 256,
    'depth': 12,
    'num_heads': 8,
    'mlp_ratio': 4.0,
    'class_dropout_prob': 0.1,
    'num_classes': 1,
    'learn_sigma': False,
    'min_inter_site_interaction_range': 1,
    'max_inter_site_interaction_range': 2,
    'input_embedder': 'hamiltonian',
    'output_unpatcher': 'hamiltonian',
    # 'on_site_block_names': ['iy1'],
    # 'inter_site_block_names': ['1z 1x iy1 iyx'],
    'on_site_block_names': ['xiy', '1iy', 'ziy'],
    'inter_site_block_names': ['1x', '1z', 'iyx'],
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

save_params(params, os.path.join(root_dir, 'params.json'))
save_params(dit_c2h_config, os.path.join(root_dir, 'dit_c2h_config.json'))
save_params(dit_h2c_config, os.path.join(root_dir, 'dit_h2c_config.json'))
save_params(dit_h2h_config, os.path.join(root_dir, 'dit_h2h_config.json'))

loss_path = os.path.join(root_dir, loss_file)
convergence_path = os.path.join(root_dir, convergence_file)

tests_sub_path = os.path.join(root_dir, tests_sub_dir)     
if not os.path.isdir(tests_sub_path):
    os.makedirs(tests_sub_path)

# Try to load data statistics
try:
    with open(normalization_params_path, 'rb') as f:
        normalization_params = pickle.load(f)
except:
    data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down, conductance_config=conductance_config, random_noise=params['random_noise'])
    data_loader = DataLoader(data, params['batch_size'])
    normalization_params = calculate_mean_and_std(data_loader, device=device)
    with open(normalization_params_path, 'wb') as f:
        pickle.dump(normalization_params, f)

print('Normalization params:', normalization_params)

# data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down, conductance_config=conductance_config)
data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_params=normalization_params, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down, conductance_config=conductance_config, random_noise=params['random_noise'])

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])# train_data = Subset(data, range(len(data) // 2, len(data) // 2 + 1))

# sampler = RandomSampler(train_data, replacement=True, num_samples=1000)
train_loader = DataLoader(data, params['batch_size']) #, sampler=sampler)
test_loader = DataLoader(test_data, params['batch_size'])

dit_c2h_config = load_params(os.path.join(pretrained_model_path, 'dit_c2h_config.json'))
dit_h2c_config = load_params(os.path.join(pretrained_model_path, 'dit_h2c_config.json'))

model_c2h = load_model(DiT, dit_c2h_config, pretrained_model_path, epoch=pretrained_model_epoch, suffix='_c2h')
model_h2c = load_model(DiT, dit_h2c_config, pretrained_model_path, epoch=pretrained_model_epoch, suffix='_h2c')

# model_c2h = DiT(**dit_c2h_config)
optimizer_c2h = torch.optim.Adam(model_c2h.parameters(), lr=params['pretrained_lr'])
scheduler_c2h = torch.optim.lr_scheduler.MultiStepLR(optimizer_c2h, milestones=[100, 400, 800])

# model_h2c = DiT(**dit_h2c_config)
optimizer_h2c = torch.optim.Adam(model_h2c.parameters(), lr=params['pretrained_lr'])
scheduler_h2c = torch.optim.lr_scheduler.MultiStepLR(optimizer_h2c, milestones=[100, 400, 800])

model_h2h = DiT(**dit_h2h_config)
optimizer_h2h = torch.optim.Adam(model_h2h.parameters(), lr=params['lr'])
scheduler_h2h = torch.optim.lr_scheduler.MultiStepLR(optimizer_h2h, milestones=[50, 100, 200, 400, 600, 800])

save_data_list(['Epoch', 'C2H Training loss', "C2H Test loss", 'H2C Training loss', "H2C Test loss", "H2C Test loss on C2H output", "H2H Training loss", "H2H Test loss", "H2C Test loss on H2H output"], loss_path, mode='w')

h_mean, h_std = normalization_params[0]
h_denormalize = Denormalize(mean=h_mean, std=h_std)

cmap_mean, cmap_std = normalization_params[1]
cmap_normalize = Normalize(mean=cmap_mean, std=cmap_std)
cmap_denormalize = Denormalize(mean=cmap_mean, std=cmap_std)

noisy_cmap_mean, noisy_cmap_std = normalization_params[3]
noisy_cmap_denormalize = Denormalize(mean=noisy_cmap_mean, std=noisy_cmap_std)

for epoch in range(0, params['epochs'] + 1):
    train_loss_c2h, train_loss_h2c, train_loss_h2h = train_denoising_conductance_3models(
        model_c2h,
        model_h2c,
        model_h2h,
        train_loader,
        optimizer_c2h,
        optimizer_h2c,
        optimizer_h2h,
        device,
        epoch,
        h_denormalize,
        cmap_normalize,
        cmap_config=conductance_config,
        model_prediction=params['model_prediction'],
        h2h_start_epoch=params['h2h_start_epoch'],
        train_c2h=params['train_c2h'],
        train_h2c=params['train_h2c']
    )
    test_loss_c2h, test_loss_h2c, test_loss_h2c_on_c2h, test_loss_h2h, test_loss_h2c_on_h2h = test_denoising_conductance_3models(
        model_c2h,
        model_h2c,
        model_h2h,
        test_loader,
        device,
        epoch,
        h_denormalize,
        cmap_normalize,
        cmap_config=conductance_config,
        model_prediction=params['model_prediction']
    )
    scheduler_c2h.step()
    scheduler_h2c.step()
    scheduler_h2h.step()
    save_data_list([epoch, train_loss_c2h, test_loss_c2h, train_loss_h2c, test_loss_h2c, test_loss_h2c_on_c2h, train_loss_h2h, test_loss_h2h, test_loss_h2c_on_h2h], loss_path)
    
    # sample and save every 10 epochs
    if epoch % 10 == 0:
        save_model(model_c2h, root_dir, epoch, suffix='_c2h')
        save_model(model_h2c, root_dir, epoch, suffix='_h2c')
        save_model(model_h2h, root_dir, epoch, suffix='_h2h')

        model_c2h.eval()
        model_c2h.to(device)
        epoch_dir = os.path.join(tests_sub_path, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        eigvals_test_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'reference'))
        test_sample = test_data[0]
        h_torch_normalized = test_sample[0][0].unsqueeze(0).to(device)
        h_torch_denormalized = h_denormalize(h_torch_normalized)[0]
        test_hamiltonian = TorchHamiltonian.from_2channel_tensor(h_torch_denormalized)
        plot_eigvals_levels(test_hamiltonian, save_path=eigvals_test_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)

        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_torch_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('ref_real'), vmin=-vscale, vmax=vscale)
        plot_matrix(h_torch_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('ref_imag'), vmin=-vscale, vmax=vscale)
        
        eigvals_dit_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'DiT_rec'))
        eigvals_noisy_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'noisy'))

        h_noisy_conductance = test_sample[0][3].unsqueeze(0).to(device)

        mock_noise_amplitude = torch.zeros(1, 1).to(device)
        real_noise_amplitude = torch.tensor(test_sample[1][1]).to(device).view(1, 1)

        h_predicted = model_c2h(h_noisy_conductance, mock_noise_amplitude, None)
        
        h_predicted_denorm = h_denormalize(h_predicted)[0]
        
        ham_rec = TorchHamiltonian.from_2channel_tensor(h_predicted_denorm)
        plot_eigvals_levels(ham_rec, save_path=eigvals_dit_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)
        
        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_predicted_denorm[0].detach().cpu().numpy(), test_matrix_path.format('reconstructed_real'), vmin=-vscale, vmax=vscale)
        plot_matrix(h_predicted_denorm[1].detach().cpu().numpy(), test_matrix_path.format('reconstructed_imag'), vmin=-vscale, vmax=vscale)
        
        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            noisy_conductance_path = os.path.join(epoch_dir, 'noisy_conductance')
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
            ref_conductance_path = os.path.join(epoch_dir, 'ref_conductance')
            os.makedirs(ref_conductance_path, exist_ok=True)
            x_tick_range = cmap_config['cmap2']['b_range']
            y_tick_range = cmap_config['cmap2']['ef_range']
            i_val = cmap_config['cmap2']['i']
            j_val = cmap_config['cmap2']['j']
            ref_cmap = test_sample[0][1].to(device)
            ref_cmap = cmap_denormalize(ref_cmap)
            plot_conductance_map(
                ref_cmap[i].detach().cpu().numpy(),
                os.path.join(ref_conductance_path, f'map_i{i_val}_j{j_val}.png'),
                xtick_range=x_tick_range,
                ytick_range=y_tick_range,
                xlabel="$B$ [mV]",
                ylabel="$E_F$ (meV)"
            )

        mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(h_predicted_denorm[0], h_predicted_denorm[1]))
        predicted_cmap = generate_conductance_tensor(mapped_h_predicted, conductance_config)
        
        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            predicted_conductance_path = os.path.join(epoch_dir, 'predicted_conductance')
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


        # H2H evaluation
        model_h2h.eval()
        model_h2h.to(device)
        eigvals_dit_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'DiT_imp'))

        improved_h = model_h2h(h_predicted, real_noise_amplitude, None)
        if params['model_prediction'] == 'hamiltonian_improvement_mask':
            multiplication_factor = abs(improved_h).clone()
            improved_h = torch.where(multiplication_factor > 0, multiplication_factor * h_predicted, h_predicted)

        improved_h_denorm = h_denormalize(improved_h)[0]
        
        ham_rec = TorchHamiltonian.from_2channel_tensor(improved_h_denorm)
        plot_eigvals_levels(ham_rec, save_path=eigvals_dit_path, representation_mapping=RepresentationMapping.none, ylim=ylim, ynorm=ynorm)
        
        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(improved_h_denorm[0].detach().cpu().numpy(), test_matrix_path.format('improved_real'), vmin=-vscale, vmax=vscale)
        plot_matrix(improved_h_denorm[1].detach().cpu().numpy(), test_matrix_path.format('improved_imag'), vmin=-vscale, vmax=vscale)
    
        mapped_h_predicted_h2h = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(improved_h_denorm[0], improved_h_denorm[1]))
        h2h_predicted_cmap = generate_conductance_tensor(mapped_h_predicted_h2h, conductance_config)

        # h2h_predicted_cmap = torch.clip(h2h_predicted_cmap, -2., 2.)
        
        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            h2h_conductance_path = os.path.join(epoch_dir, 'improved_conductance')
            os.makedirs(h2h_conductance_path, exist_ok=True)
            x_tick_range = cmap_config['cmap2']['b_range']
            y_tick_range = cmap_config['cmap2']['ef_range']
            i_val = cmap_config['cmap2']['i']
            j_val = cmap_config['cmap2']['j']
            plot_conductance_map(
                h2h_predicted_cmap[i].detach().cpu().numpy(),
                os.path.join(h2h_conductance_path, f'map_i{i_val}_j{j_val}.png'),
                xtick_range=x_tick_range,
                ytick_range=y_tick_range,
                xlabel="$B$ [mV]",
                ylabel="$E_F$ (meV)"
            )

        # H2C evaluation
        model_h2c.eval()
        model_h2c.to(device)

        cmap = model_h2c(h_predicted, mock_noise_amplitude, None)
        cmap = torch.cat((cmap, torch.zeros((cmap.shape[0], 2, *cmap.shape[2:]), device=device)), dim=1)
        cmap_denorm = cmap_denormalize(cmap)[0]

        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            cmap_h2c_conductance_path = os.path.join(epoch_dir, 'h2c_from_c2h_conductance')
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

        
        cmap = model_h2c(improved_h, mock_noise_amplitude, None)
        cmap = torch.cat((cmap, torch.zeros((cmap.shape[0], 2, *cmap.shape[2:]), device=device)), dim=1)
        cmap_denorm = cmap_denormalize(cmap)[0]

        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            cmap_h2c_conductance_path = os.path.join(epoch_dir, 'h2c_from_h2h_conductance')
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
            cmap_h2c_conductance_path = os.path.join(epoch_dir, 'h2c_from_ref_conductance')
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

plot_convergence(loss_path, convergence_path, read_label=True)
