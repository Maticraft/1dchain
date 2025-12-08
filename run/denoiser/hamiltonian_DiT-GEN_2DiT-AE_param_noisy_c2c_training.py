import json
import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader, Subset, RandomSampler
import torch
from torchvision.transforms import Normalize

from src.data.datasets import HamiltonianFromParametersDataset
from src.data.utils import calculate_mean_and_std, Denormalize
from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.conductance import plot_conductance_map, generate_conductance_tensor
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.hamiltonian.hamiltonian_torch_handlers import HamiltonianParams, HamiltonianConverter
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonian
from src.hamiltonian.utils import majoranization, plot_eigvals_levels
from src.models.denoiser import train_denoising_conductance_DiTGEN_AE, test_denoising_conductance_DiTGEN_AE
from src.models.files import save_params, save_model, save_data_list, load_params, load_model
from src.models.diffusion_transformer import DiT
from src.models.utils import deep_update, weighted_update, tensor_dict_to_list
from src.plots import plot_convergence, plot_matrix
from src.hamiltonian.utils import plot_eigvals_levels, plot_eigvals
from src.hamiltonian.torch_hamiltonian import TorchHamiltonian


# TODO: Calculate majoranization in 3 points (mu +0.1 mu -0.1 eV) -> loss must denormalized

# Paths
data_path = './data/quantum_dots/simple_fixed_t_3dots1level_majoranization0_0_mzm_gap0_25'
normalization_params_path = f'{data_path}/std_rep_normalization_params_1b-3mu-maps.pkl'
save_dir = './conductance/quantum_dots/simple_fixed_t_3dots1level_majoranization0_0_mzm_gap0_25'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_majoranas_latent_ep_{}'
eigvals_plot_name = 'eigvals_spectre_{}.png'
hamiltonian_plot_name = 'hamiltonian_{}.png'

# Reference eigvals plot params
tests_sub_dir = 'tests'
x_axis = 'potential'
x_values = np.linspace(-.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-2., 2.)
vscale = 1/AtomicUnits.Eh

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model name
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-params_majorana-destructing-noise1-extra-params_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'
model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-b-const-params_majorana-destructing-noise1-extra-params_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'

# Pretrained model
# pretrained_model_name = 'QDH-1lvl-no-interlevel_2DiTs_real_12_patch1_4maps50x50embed_norm_triple-c2c_lr1e-4decreasing'
# pretrained_model_path = f'./conductance/quantum_dots/3dots1level_majoranas_gap_pol_verified/DiT/{pretrained_model_name}'
# pretrained_model_epoch = 350

# Params
params = {
    'epochs': 500,
    'batch_size': 16,
    # 'pretrained_lr': 1e-4,
    'lr': 1e-4,
    'max_noise_amplitude': 1.,
    'random_noise': True,
    'train_gen': True,
    'train_ae': True,
    'n_dots': 3,
    'reg_strength': 0.1,
    'site_independent_noise': True,  # If False, the noise is applied to all sites equally
}


# Architecture
dit_c2h_config = {
    'input_size': 50,
    'input_channels': 4,
    'output_size': 12,
    'output_channels': 2,
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
    # 'output_hamiltonian_params': HamiltonianParams(),
    "output_hamiltonian_params": HamiltonianParams(
        on_site_real_params=("potential", "magnetic_field"),
        on_site_imag_params=(()),
        inter_site_real_params=(()),
        # inter_site_real_params=("t", "t_phase_diag", "t_phase_antidiag"),
        inter_site_imag_params=(()),
        site_constant_params=("magnetic_field",),
    )    
    # 'on_site_block_names': ['iy1'],
    # 'inter_site_block_names': ['1z 1x iy1 iyx'],
}

dit_encoder_config = {
    'input_size': 12,
    'input_channels': 2,
    'output_size': 4,
    'output_channels': 1,
    'patch_size': 1,
    'hidden_size': 256,
    'depth': 6,
    'num_heads': 4,
    'mlp_ratio': 4.0,
    'class_dropout_prob': 0.1,
    'num_classes': 1,
    'learn_sigma': False,
    'min_inter_site_interaction_range': 1,
    'max_inter_site_interaction_range': 2,
    'input_embedder': 'hamiltonian',
    'output_unpatcher': 'patch',
    'input_hamiltonian_params': HamiltonianParams(
        inter_site_real_params=("t", "t_phase_diag", "t_phase_antidiag"),
        inter_site_imag_params=(()),
    ),
    # 'output_hamiltonian_params': HamiltonianParams()
    # 'on_site_block_names': ['iy1'],
    # 'inter_site_block_names': ['1z 1x iy1 iyx'],
}

dit_decoder_config = {
    'input_size': 4,
    'input_channels': 1,
    'output_size': 12,
    'output_channels': 2,
    'patch_size': 1,
    'hidden_size': 256,
    'depth': 6,
    'num_heads': 4,
    'mlp_ratio': 4.0,
    'class_dropout_prob': 0.1,
    'num_classes': 1,
    'learn_sigma': False,
    'min_inter_site_interaction_range': 1, # commenting this line breaks the generated hamiltonian
    'max_inter_site_interaction_range': 2,
    'input_embedder': 'patch',
    'output_unpatcher': 'hamiltonian',
    # 'input_hamiltonian_params': HamiltonianParams(),
    'output_hamiltonian_params': HamiltonianParams(
        inter_site_real_params=("t", "t_phase_diag", "t_phase_antidiag"),
        inter_site_imag_params=(()),
    ),
    # 'on_site_block_names': ['iy1'],
    # 'inter_site_block_names': ['1z 1x iy1 iyx'],
}


hamiltonian_converter = HamiltonianConverter(
    hamiltonian_params=dit_encoder_config['input_hamiltonian_params'],
    min_inter_site_interaction_range=dit_encoder_config['min_inter_site_interaction_range'],
    max_inter_site_interaction_range=dit_encoder_config['max_inter_site_interaction_range'],
)

defaults = DefaultParameters()


conductance_config = {
    'cmap_list': [
    {
        'cmap2': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'b_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
            'b_num': 50,
            'ef_num': 50,
            'with_embedding': False
        },
    },
    {
        'cmap0': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'mu_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
            'mu_num': 50,
            'ef_num': 50,
            'site_index': 1,
            'with_embedding': False,
        },
    },
    {
        'cmap0': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'mu_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
            'mu_num': 50,
            'ef_num': 50,
            'site_index': 0,
            'with_embedding': False,
        },
    },
    {
        'cmap0': {
            'i':0,
            'j':0,
            'gamma': 0.1,
            'mu_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
            'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
            'mu_num': 50,
            'ef_num': 50,
            'site_index': 2,
            'with_embedding': False
        },
    },
    ],
}

# conductance_config = {
#     'cmap_list': [
#     {
#         'cmap2': {
#             'i':0,
#             'j':0,
#             'gamma': 0.1,
#             'b_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
#             'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
#             'b_num': 50,
#             'ef_num': 50,
#             'with_embedding': False
#         },
#     },
#     {
#         'cmap2': {
#             'i':0,
#             'j':1,
#             'gamma': 0.1,
#             'b_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
#             'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
#             'b_num': 50,
#             'ef_num': 50,
#             'with_embedding': False
#         },
#     },
#         {
#         'cmap2': {
#             'i':1,
#             'j':0,
#             'gamma': 0.1,
#             'b_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
#             'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
#             'b_num': 50,
#             'ef_num': 50,
#             'with_embedding': False
#         },
#     },
#     {
#         'cmap2': {
#             'i':1,
#             'j':1,
#             'gamma': 0.1,
#             'b_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
#             'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
#             'b_num': 50,
#             'ef_num': 50,
#             'with_embedding': False
#         },
#     },
#     ],
# }

# Set the root dir
root_dir = os.path.join(save_dir, f'DiT', model_name)
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

save_params(params, os.path.join(root_dir, 'params.json'))
save_params(dit_c2h_config, os.path.join(root_dir, 'dit_c2h_config.json'))
save_params(dit_encoder_config, os.path.join(root_dir, 'dit_encoder_config.json'))
save_params(dit_decoder_config, os.path.join(root_dir, 'dit_decoder_config.json'))

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
    data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, conductance_config=conductance_config, random_noise=False, assert_noise_majoranas_destruction=True, n_dots=params['n_dots'])
    data_loader = DataLoader(data, params['batch_size'])
    normalization_params = calculate_mean_and_std(data_loader, device=device)
    normalization_params = [
        normalization_params[0],
        normalization_params[1],
        normalization_params[0], # noisy h has the same normalization as the original h
        normalization_params[1], # noisy cmap has the same normalization as the original cmap
    ]
    with open(normalization_params_path, 'wb') as f:
        pickle.dump(normalization_params, f)

print('Normalization params:', normalization_params)

# dit_c2h_config = load_params(os.path.join(pretrained_model_path, 'dit_c2h_config.json'))
# dit_h2c_config = load_params(os.path.join(pretrained_model_path, 'dit_h2c_config.json'))

# model_c2h = load_model(DiT, dit_c2h_config, pretrained_model_path, epoch=pretrained_model_epoch, suffix='_c2h')
# model_h2c = load_model(DiT, dit_h2c_config, pretrained_model_path, epoch=pretrained_model_epoch, suffix='_h2c')

model_c2h = DiT(**dit_c2h_config)
optimizer_c2h = torch.optim.Adam(model_c2h.parameters(), lr=params['lr'])
scheduler_c2h = torch.optim.lr_scheduler.MultiStepLR(optimizer_c2h, milestones=[50, 100, 200, 400, 600, 800])

model_encoder = DiT(**dit_encoder_config)
optimizer_encoder = torch.optim.Adam(model_encoder.parameters(), lr=params['lr'])
scheduler_encoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_encoder, milestones=[50, 100, 200, 400, 600, 800])

model_decoder = DiT(**dit_decoder_config)
optimizer_decoder = torch.optim.Adam(model_decoder.parameters(), lr=params['lr'])
scheduler_decoder = torch.optim.lr_scheduler.MultiStepLR(optimizer_decoder, milestones=[50, 100, 200, 400, 600, 800])

save_data_list(['Epoch', 'GEN Training loss', "GEN Test loss", 'AE Training loss', 'AE Contrastive training loss', "AE Test loss", "AE on GEN test loss"], loss_path, mode='w')

h_mean, h_std = normalization_params[0]
h_denormalize = Denormalize(mean=h_mean, std=h_std)

cmap_mean, cmap_std = normalization_params[1]
cmap_normalize = Normalize(mean=cmap_mean, std=cmap_std)
cmap_denormalize = Denormalize(mean=cmap_mean, std=cmap_std)

noisy_cmap_mean, noisy_cmap_std = normalization_params[3]
noisy_cmap_denormalize = Denormalize(mean=noisy_cmap_mean, std=noisy_cmap_std)

for epoch in range(0, params['epochs'] + 1):
    l = params['max_noise_amplitude'] # * epoch / params['epochs']
    params_noise_config = {
        'parameters': {
            "mu": (l, defaults.mu_range[1]),
            "t": (l, defaults.t_range[1]),
            "b": (l, defaults.b_range[1]),
            "d": (l, defaults.d_range[1]),
            # "ph_d": (l, defaults.ph_d_range[1]),
            "l": (l, defaults.l_range[1]),
            # "l_rho": (l, defaults.l_rho_range[1]),
            # "l_ksi": (l, defaults.l_ksi_range[1]),
        }
    }

    # data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_mean=mean, normalization_std=std, representation_mapping=RepresentationMapping.majorana_plus_minus_up_down, conductance_config=conductance_config)
    data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_params=normalization_params, conductance_config=conductance_config, random_noise=params['random_noise'], assert_noise_majoranas_destruction=True, n_dots=params['n_dots'], site_independent_noise=params['site_independent_noise'])

    train_size = int(0.99*len(data))
    test_size = len(data) - train_size

    train_data, test_data = random_split(data, [train_size, test_size])# train_data = Subset(data, range(len(data) // 2, len(data) // 2 + 1))

    # sampler = RandomSampler(train_data, replacement=True, num_samples=1000)
    train_loader = DataLoader(data, params['batch_size']) #, sampler=sampler)
    test_loader = DataLoader(test_data, params['batch_size'])


    train_loss_gen, train_loss_ae, train_loss_contrastive = train_denoising_conductance_DiTGEN_AE(
        model_c2h,
        model_encoder,
        model_decoder,
        hamiltonian_converter,
        train_loader,
        optimizer_c2h,
        optimizer_encoder,
        optimizer_decoder,
        device,
        epoch,
        train_gen=params['train_gen'],
        train_ae=params['train_ae'],
        regularization_strength=params['reg_strength'],
    )
    test_loss_gen, test_loss_ae, test_loss_ae_on_gen = test_denoising_conductance_DiTGEN_AE(
        model_c2h,
        model_encoder,
        model_decoder,
        hamiltonian_converter,
        test_loader,
        device,
        epoch,
    )
    scheduler_c2h.step()
    scheduler_encoder.step()
    save_data_list([epoch, train_loss_gen, test_loss_gen, train_loss_ae, train_loss_contrastive, test_loss_ae, test_loss_ae_on_gen], loss_path)
    
    # sample and save every 10 epochs
    if epoch % 10 == 0:
        save_model(model_c2h, root_dir, epoch, suffix='_c2h')
        save_model(model_encoder, root_dir, epoch, suffix='_encoder')
        save_model(model_decoder, root_dir, epoch, suffix='_decoder')

        model_c2h.eval()
        model_c2h.to(device)
        epoch_dir = os.path.join(tests_sub_path, f'epoch_{epoch}')
        os.makedirs(epoch_dir, exist_ok=True)

        test_sample = test_data[0]
        h_torch_normalized = test_sample[0][0].unsqueeze(0).to(device)
        h_torch_denormalized = h_denormalize(h_torch_normalized)[0]
        h_complex_tensor = torch.complex(h_torch_denormalized[0], h_torch_denormalized[1])
        h_label = majoranization(h_complex_tensor.detach().cpu().numpy(), params["n_dots"])

        h_map = hamiltonian_converter.from_matrix_to_params(h_torch_denormalized.unsqueeze(0))
        h_map_json = tensor_dict_to_list(h_map)
        with open(os.path.join(epoch_dir, 'h_ref_map.json'), 'w') as f:
            json.dump(h_map_json, f, indent=4)

        test_hamiltonian = TorchHamiltonian.from_2channel_tensor_with_params(h_torch_denormalized, dit_encoder_config['input_hamiltonian_params'])

        eigvals_occ_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'occ_reference'))
        plot_eigvals(test_hamiltonian, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_label}', majoranization=True)
        eigvals_eh_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'eh_reference'))
        plot_eigvals(test_hamiltonian, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_label}')

        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_torch_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('ref_real'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_label}')
        plot_matrix(h_torch_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('ref_imag'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_label}')
        
        h_perturbed = test_sample[0][2].unsqueeze(0).to(device)
        h_noisy_denormalized = h_denormalize(h_perturbed)[0]
        h_complex_tensor = torch.complex(h_noisy_denormalized[0], h_noisy_denormalized[1])
        h_noisy_label = majoranization(h_complex_tensor.detach().cpu().numpy(), params["n_dots"])
        
        h_noisy_map = hamiltonian_converter.from_matrix_to_params(h_noisy_denormalized.unsqueeze(0))
        h_noisy_map_json = tensor_dict_to_list(h_noisy_map)
        with open(os.path.join(epoch_dir, 'h_noisy_map.json'), 'w') as f:
            json.dump(h_noisy_map_json, f, indent=4)

        test_noisy_hamiltonian = TorchHamiltonian.from_2channel_tensor_with_params(h_noisy_denormalized, dit_encoder_config['input_hamiltonian_params'])
        eigvals_occ_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'occ_noisy'))
        plot_eigvals(test_noisy_hamiltonian, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_noisy_label}', majoranization=True)
        eigvals_eh_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'eh_noisy'))
        plot_eigvals(test_noisy_hamiltonian, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_noisy_label}')

        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_noisy_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('noisy_real'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_noisy_label}')
        plot_matrix(h_noisy_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('noisy_imag'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_noisy_label}')
        
        mock_noise_amplitude = torch.zeros(1, 1).to(device)
        real_noise_amplitude = torch.tensor(test_sample[1][1]).to(device).view(1, 1)

        h_perturbed_params_map = hamiltonian_converter.from_matrix_to_params(h_perturbed)

        h_noisy_conductance = test_sample[0][3].unsqueeze(0).to(device)
        params_map = model_c2h(h_noisy_conductance, real_noise_amplitude, None, matrix_output=False)
        improved_map = deep_update(h_perturbed_params_map, params_map, detach=False)

        # improved_map = weighted_update(h_perturbed_params_map, params_map, alpha=0.5)
        # improved_map = params_map

        h_predicted = hamiltonian_converter.from_params_to_matrix(improved_map)
        h_predicted_denorm = h_denormalize(h_predicted)[0]

        h_predicted_map = hamiltonian_converter.from_matrix_to_params(h_predicted_denorm.unsqueeze(0))
        h_predicted_map_json = tensor_dict_to_list(h_predicted_map)
        with open(os.path.join(epoch_dir, 'h_predicted_map.json'), 'w') as f:
            json.dump(h_predicted_map_json, f, indent=4)

        h_complex_tensor = torch.complex(h_predicted_denorm[0], h_predicted_denorm[1])
        h_predicted_label = majoranization(h_complex_tensor.detach().cpu().numpy(), params["n_dots"])

        eigvals_dit_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'GEN_rec'))

        ham_rec = TorchHamiltonian.from_2channel_tensor_with_params(h_predicted_denorm, dit_encoder_config['input_hamiltonian_params'])
        eigvals_occ_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'occ_GEN_rec'))
        plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_predicted_label}', majoranization=True)
        eigvals_eh_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'eh_GEN_rec'))
        plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_predicted_label}')
        
        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_predicted_denorm[0].detach().cpu().numpy(), test_matrix_path.format('GEN_improved_real'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_predicted_label}')
        plot_matrix(h_predicted_denorm[1].detach().cpu().numpy(), test_matrix_path.format('GEN_improved_imag'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_predicted_label}')
        
        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            noisy_conductance_path = os.path.join(epoch_dir, 'noisy_conductance')
            os.makedirs(noisy_conductance_path, exist_ok=True)
            cmap_name = list(cmap_config.keys())[0]
            x_name = 'b' if cmap_name == 'cmap2' else 'mu'

            x_tick_range = cmap_config[cmap_name][f'{x_name}_range']
            y_tick_range = cmap_config[cmap_name]['ef_range']
            i_val = cmap_config[cmap_name]['i']
            j_val = cmap_config[cmap_name]['j']
            if cmap_name == 'cmap0':
                filename_suffix = f'_dot{cmap_config[cmap_name]["site_index"]}'
            else:
                filename_suffix = ''

            h_noisy_conductance = noisy_cmap_denormalize(h_noisy_conductance)
            plot_conductance_map(
                h_noisy_conductance[0, i].detach().cpu().numpy(),
                os.path.join(noisy_conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{filename_suffix}.png'),
                xtick_range=x_tick_range,
                ytick_range=y_tick_range,
                xlabel=f"${x_name}$ [mV]",
                ylabel="$E_F$ (meV)",
                title=f'Noise amplitude: {real_noise_amplitude.item():.2f}, Majoranization: {h_noisy_label}'
            )

        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            ref_conductance_path = os.path.join(epoch_dir, 'ref_conductance')
            os.makedirs(ref_conductance_path, exist_ok=True)
            cmap_name = list(cmap_config.keys())[0]
            x_name = 'b' if cmap_name == 'cmap2' else 'mu'

            x_tick_range = cmap_config[cmap_name][f'{x_name}_range']
            y_tick_range = cmap_config[cmap_name]['ef_range']
            i_val = cmap_config[cmap_name]['i']
            j_val = cmap_config[cmap_name]['j']
            ref_cmap = test_sample[0][1].to(device)
            ref_cmap = cmap_denormalize(ref_cmap)

            if cmap_name == 'cmap0':
                filename_suffix = f'_dot{cmap_config[cmap_name]["site_index"]}'
            else:
                filename_suffix = ''

            plot_conductance_map(
                ref_cmap[i].detach().cpu().numpy(),
                os.path.join(ref_conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{filename_suffix}.png'),
                xtick_range=x_tick_range,
                ytick_range=y_tick_range,
                xlabel=f"${x_name}$ [mV]",
                ylabel="$E_F$ (meV)",
                title=f'Majoranization: {h_label}'
            )

        mapped_h_predicted = torch.complex(h_predicted_denorm[0], h_predicted_denorm[1])
        predicted_cmap = generate_conductance_tensor(mapped_h_predicted, conductance_config)
        
        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            predicted_conductance_path = os.path.join(epoch_dir, 'GEN_improved_conductance')
            os.makedirs(predicted_conductance_path, exist_ok=True)
            cmap_name = list(cmap_config.keys())[0]
            x_name = 'b' if cmap_name == 'cmap2' else 'mu'

            x_tick_range = cmap_config[cmap_name][f'{x_name}_range']
            y_tick_range = cmap_config[cmap_name]['ef_range']
            i_val = cmap_config[cmap_name]['i']
            j_val = cmap_config[cmap_name]['j']

            if cmap_name == 'cmap0':
                filename_suffix = f'_dot{cmap_config[cmap_name]["site_index"]}'
            else:
                filename_suffix = ''

            plot_conductance_map(
                predicted_cmap[i].detach().cpu().numpy(),
                os.path.join(predicted_conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{filename_suffix}.png'),
                xtick_range=x_tick_range,
                ytick_range=y_tick_range,
                xlabel=f"${x_name}$ [mV]",
                ylabel="$E_F$ (meV)",
                title=f'Majoranization: {h_predicted_label}'
            )


        # AE evaluation
        model_encoder.eval()
        model_encoder.to(device)

        model_decoder.eval()
        model_decoder.to(device)

        # Reference
        latent_h = model_encoder(h_torch_normalized, mock_noise_amplitude, None)
        reconstructed_h = model_decoder(latent_h, mock_noise_amplitude, None)

        h_rec_denorm = h_denormalize(reconstructed_h)[0]
        h_complex_tensor = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
        h_rec_label = majoranization(h_complex_tensor.detach().cpu().numpy(), params["n_dots"])
        

        ham_rec = TorchHamiltonian.from_2channel_tensor_with_params(h_rec_denorm, dit_encoder_config['input_hamiltonian_params'])
        eigvals_occ_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'occ_AE_rec'))
        plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_rec_label}', majoranization=True)
        eigvals_eh_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'eh_AE_rec'))
        plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_rec_label}')
                
        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_rec_denorm[0].detach().cpu().numpy(), test_matrix_path.format('AE_reconstructed_real'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_rec_label}')
        plot_matrix(h_rec_denorm[1].detach().cpu().numpy(), test_matrix_path.format('AE_reconstructed_imag'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_rec_label}')

        mapped_h_predicted = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
        cmap_rec = generate_conductance_tensor(mapped_h_predicted, conductance_config)
        
        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            predicted_conductance_path = os.path.join(epoch_dir, 'AE_reconstructed_conductance')
            os.makedirs(predicted_conductance_path, exist_ok=True)
            cmap_name = list(cmap_config.keys())[0]
            x_name = 'b' if cmap_name == 'cmap2' else 'mu'

            x_tick_range = cmap_config[cmap_name][f'{x_name}_range']
            y_tick_range = cmap_config[cmap_name]['ef_range']
            i_val = cmap_config[cmap_name]['i']
            j_val = cmap_config[cmap_name]['j']

            if cmap_name == 'cmap0':
                filename_suffix = f'_dot{cmap_config[cmap_name]["site_index"]}'
            else:
                filename_suffix = ''

            plot_conductance_map(
                cmap_rec[i].detach().cpu().numpy(),
                os.path.join(predicted_conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{filename_suffix}.png'),
                xtick_range=x_tick_range,
                ytick_range=y_tick_range,
                xlabel=f"${x_name}$ [mV]",
                ylabel="$E_F$ (meV)",
                title=f'Majoranization: {h_rec_label}'
            )

        # Noisy 
        latent_h = model_encoder(h_perturbed, mock_noise_amplitude, None)
        reconstructed_h = model_decoder(latent_h, mock_noise_amplitude, None)

        h_rec_denorm = h_denormalize(reconstructed_h)[0]
        h_complex_tensor = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
        h_rec_noisy_label = majoranization(h_complex_tensor.detach().cpu().numpy(), params["n_dots"])
    
        ham_rec = TorchHamiltonian.from_2channel_tensor_with_params(h_rec_denorm, dit_encoder_config['input_hamiltonian_params'])
        eigvals_occ_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'occ_AE_rec_noisy'))
        plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_rec_noisy_label}', majoranization=True)
        eigvals_eh_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'eh_AE_rec_noisy'))
        plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_rec_noisy_label}')
                
        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_rec_denorm[0].detach().cpu().numpy(), test_matrix_path.format('AE_reconstructed_noisy_real'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_rec_noisy_label}')
        plot_matrix(h_rec_denorm[1].detach().cpu().numpy(), test_matrix_path.format('AE_reconstructed_noisy_imag'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_rec_noisy_label}')

        mapped_h_predicted = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
        cmap_rec = generate_conductance_tensor(mapped_h_predicted, conductance_config)
        
        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            predicted_conductance_path = os.path.join(epoch_dir, 'AE_reconstructed_noisy_conductance')
            os.makedirs(predicted_conductance_path, exist_ok=True)
            cmap_name = list(cmap_config.keys())[0]
            x_name = 'b' if cmap_name == 'cmap2' else 'mu'

            x_tick_range = cmap_config[cmap_name][f'{x_name}_range']
            y_tick_range = cmap_config[cmap_name]['ef_range']
            i_val = cmap_config[cmap_name]['i']
            j_val = cmap_config[cmap_name]['j']

            if cmap_name == 'cmap0':
                filename_suffix = f'_dot{cmap_config[cmap_name]["site_index"]}'
            else:
                filename_suffix = ''

            plot_conductance_map(
                cmap_rec[i].detach().cpu().numpy(),
                os.path.join(predicted_conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{filename_suffix}.png'),
                xtick_range=x_tick_range,
                ytick_range=y_tick_range,
                xlabel=f"${x_name}$ [mV]",
                ylabel="$E_F$ (meV)",
                title=f'Majoranization: {h_rec_noisy_label}'
            )

        # AE from GEN
        latent_h = model_encoder(h_predicted, mock_noise_amplitude, None)
        reconstructed_h = model_decoder(latent_h, mock_noise_amplitude, None)

        h_rec_denorm = h_denormalize(reconstructed_h)[0]
        h_complex_tensor = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
        h_rec_imp_label = majoranization(h_complex_tensor.detach().cpu().numpy(), params["n_dots"])
    
        ham_rec = TorchHamiltonian.from_2channel_tensor_with_params(h_rec_denorm, dit_encoder_config['input_hamiltonian_params'])
        eigvals_occ_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'occ_AE_GEN_rec'))
        plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_rec_imp_label}', majoranization=True)
        eigvals_eh_path = os.path.join(epoch_dir, eigvals_plot_name.format(f'eh_AE_GEN_rec'))
        plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, title=f'Majoranization: {h_rec_imp_label}')
                    
        test_matrix_path = os.path.join(epoch_dir, hamiltonian_plot_name)
        plot_matrix(h_rec_denorm[0].detach().cpu().numpy(), test_matrix_path.format('AE_GEN_reconstructed_real'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_rec_imp_label}')
        plot_matrix(h_rec_denorm[1].detach().cpu().numpy(), test_matrix_path.format('AE_GEN_reconstructed_imag'), vmin=-vscale, vmax=vscale, title=f'Majoranization: {h_rec_imp_label}')

        mapped_h_predicted = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
        cmap_rec = generate_conductance_tensor(mapped_h_predicted, conductance_config)
        
        for i, cmap_config in enumerate(conductance_config['cmap_list']):
            predicted_conductance_path = os.path.join(epoch_dir, 'AE_GEN_reconstructed_conductance')
            os.makedirs(predicted_conductance_path, exist_ok=True)
            cmap_name = list(cmap_config.keys())[0]
            x_name = 'b' if cmap_name == 'cmap2' else 'mu'

            x_tick_range = cmap_config[cmap_name][f'{x_name}_range']
            y_tick_range = cmap_config[cmap_name]['ef_range']
            i_val = cmap_config[cmap_name]['i']
            j_val = cmap_config[cmap_name]['j']

            if cmap_name == 'cmap0':
                filename_suffix = f'_dot{cmap_config[cmap_name]["site_index"]}'
            else:
                filename_suffix = ''

            plot_conductance_map(
                cmap_rec[i].detach().cpu().numpy(),
                os.path.join(predicted_conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{filename_suffix}.png'),
                xtick_range=x_tick_range,
                ytick_range=y_tick_range,
                xlabel=f"${x_name}$ [mV]",
                ylabel="$E_F$ (meV)",
                title=f'Majoranization: {h_rec_imp_label}'
            )    

    plot_convergence(loss_path, convergence_path, read_label=True)
