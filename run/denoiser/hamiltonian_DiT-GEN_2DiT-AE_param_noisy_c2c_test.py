import json
import os
import pickle

from copy import deepcopy

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
from src.hamiltonian.utils import majoranization, plot_eigvals, plot_eigvals_levels
from src.models.denoiser import train_denoising_conductance_DiTGEN_AE, test_denoising_conductance_DiTGEN_AE
from src.models.files import save_params, save_model, save_data_list, load_params, load_model
from src.models.diffusion_transformer import DiT
from src.models.utils import deep_update, tensor_dict_to_list, weighted_update
from src.plots import plot_convergence, plot_matrix
from src.hamiltonian.utils import plot_eigvals_levels
from src.hamiltonian.torch_hamiltonian import TorchHamiltonian

# Paths
data_path = './data/quantum_dots/simple_fixed_t_3dots1level_majoranization0_0_mzm_gap0_25'
# normalization_params_path = f'{data_path}/std_rep_normalization_params_4b-maps.pkl'
normalization_params_path = f'{data_path}/std_rep_normalization_params_1b-3mu-maps.pkl'
save_dir = './conductance/quantum_dots/simple_fixed_t_3dots1level_majoranization0_0_mzm_gap0_25'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_majoranas_latent_ep_{}'
eigvals_plot_name = 'eigvals_spectre_{}.png'
hamiltonian_plot_name = 'hamiltonian_{}.png'


num_samples = 10
epoch = 500

# Reference eigvals plot params
tests_sub_dir = f'noise1_reconstruction_tests_ep_{epoch}'
x_axis = 'potential'
x_values = np.linspace(-.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-2., 2.)
vscale = 1/AtomicUnits.Eh
n_dots = 3

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model name
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-GEN-KL-latent-loss_2DiT-AE-patch4-1_improve-selected-params_gradually-increasing-noise10_4maps50x50embed_norm-c2c_lr1e-4-decreasing'
model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg1_2DiT-AE-patch4-1_improve-selected-params_majorana-destructing-noise1_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'


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
    'reg_strength': 1.,
    'site_independent_noise': True,  # If False, the noise is applied to all sites equally
}


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


conductance_config_hq = deepcopy(conductance_config)
for cmap in conductance_config_hq['cmap_list']:
    cmap_name = list(cmap.keys())[0]
    if cmap_name == 'cmap2':
        cmap[cmap_name]['b_num'] = 100
    elif cmap_name == 'cmap0':
        cmap[cmap_name]['mu_num'] = 100
    cmap[cmap_name]['ef_num'] = 100

# Set the root dir
root_dir = os.path.join(save_dir, f'DiT', model_name)
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)


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
    data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, conductance_config=conductance_config, random_noise=params['random_noise'])
    data_loader = DataLoader(data, params['batch_size'])
    normalization_params = calculate_mean_and_std(data_loader, device=device)
    with open(normalization_params_path, 'wb') as f:
        pickle.dump(normalization_params, f)

print('Normalization params:', normalization_params)

dit_c2h_config = load_params(os.path.join(root_dir, 'dit_c2h_config.json'))
dit_encoder_config = load_params(os.path.join(root_dir, 'dit_encoder_config.json'))
dit_decoder_config = load_params(os.path.join(root_dir, 'dit_decoder_config.json'))

hamiltonian_converter = HamiltonianConverter(
    dit_encoder_config['input_hamiltonian_params'],
    min_inter_site_interaction_range=dit_encoder_config['min_inter_site_interaction_range'],
    max_inter_site_interaction_range=dit_encoder_config['max_inter_site_interaction_range'],
)

model_c2h = load_model(DiT, dit_c2h_config, root_dir, epoch=epoch, suffix='_c2h')
model_c2h.to(device)
model_c2h.eval()

model_encoder = load_model(DiT, dit_encoder_config, root_dir, epoch=epoch, suffix='_encoder')
model_encoder.to(device)
model_encoder.eval()

model_decoder = load_model(DiT, dit_decoder_config, root_dir, epoch=epoch, suffix='_decoder')
model_decoder.to(device)
model_decoder.eval()

h_mean, h_std = normalization_params[0]
h_denormalize = Denormalize(mean=h_mean, std=h_std)

cmap_mean, cmap_std = normalization_params[1]
cmap_denormalize = Denormalize(mean=cmap_mean, std=cmap_std)

noisy_cmap_mean, noisy_cmap_std = normalization_params[3]
noisy_cmap_denormalize = Denormalize(mean=noisy_cmap_mean, std=noisy_cmap_std)


l = params['max_noise_amplitude']
params_noise_config = {
    'parameters': {
        "mu": (l, defaults.mu_range[1]),
        "t": (l, defaults.t_range[1]),
        # "b": (l, defaults.b_range[1]),
        # "d": (l, defaults.d_range[1]),
        # "ph_d": (l, defaults.ph_d_range[1]),
        "l": (l, defaults.l_range[1]),
        # "l_rho": (l, defaults.l_rho_range[1]),
        # "l_ksi": (l, defaults.l_ksi_range[1]),
    }
}

data = HamiltonianFromParametersDataset(data_path, QuantumDotsHamiltonian, params_noise_config, label_idx=[1, 2], format='csr', threshold=0.05, gt_threshold=True, normalization_params=normalization_params, conductance_config=conductance_config, random_noise=params['random_noise'], assert_noise_majoranas_destruction=True, n_dots=params['n_dots'], site_independent_noise=params['site_independent_noise'])

for sample_idx in range(num_samples):    
    sample_dir = os.path.join(tests_sub_path, f'sample_{sample_idx}')
    test_sample = data[sample_idx]

    os.makedirs(sample_dir, exist_ok=True)


    eigvals_test_path = os.path.join(sample_dir, eigvals_plot_name.format(f'reference'))
    h_torch_normalized = test_sample[0][0].unsqueeze(0).to(device)
    h_torch_denormalized = h_denormalize(h_torch_normalized)[0]
    h_complex_tensor = torch.complex(h_torch_denormalized[0], h_torch_denormalized[1])
    h_label = majoranization(h_complex_tensor.detach().cpu().numpy(), n_dots)

    h_map = hamiltonian_converter.from_matrix_to_params(h_torch_denormalized.unsqueeze(0) * AtomicUnits.Eh)
    h_map_json = tensor_dict_to_list(h_map)
    with open(os.path.join(sample_dir, 'h_ref_map.json'), 'w') as f:
        json.dump(h_map_json, f, indent=4)

    test_hamiltonian = TorchHamiltonian.from_2channel_tensor_with_params(h_torch_denormalized, dit_encoder_config['input_hamiltonian_params'])

    eigvals_occ_path = os.path.join(sample_dir, eigvals_plot_name.format(f'occ_reference'))
    plot_eigvals(test_hamiltonian, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', ylim=ylim, majoranization=True, xnorm=xnorm, ynorm=ynorm)
    eigvals_eh_path = os.path.join(sample_dir, eigvals_plot_name.format(f'eh_reference'))
    plot_eigvals(test_hamiltonian, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', ylim=ylim, majoranization=True, xnorm=xnorm, ynorm=ynorm)

    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_torch_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('ref_real'), vmin=-vscale, vmax=vscale, title=f'M: {h_label:.2f}')
    plot_matrix(h_torch_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('ref_imag'), vmin=-vscale, vmax=vscale, title=f'M: {h_label:.2f}')
    

    eigvals_noisy_path = os.path.join(sample_dir, eigvals_plot_name.format(f'noisy'))
    h_perturbed = test_sample[0][2].unsqueeze(0).to(device)
    h_noisy_denormalized = h_denormalize(h_perturbed)[0]
    h_complex_tensor = torch.complex(h_noisy_denormalized[0], h_noisy_denormalized[1])    
    h_noisy_label = majoranization(h_complex_tensor.detach().cpu().numpy(), n_dots)

    h_noisy_conductance_hq = generate_conductance_tensor(h_complex_tensor, conductance_config_hq)

    h_noisy_map = hamiltonian_converter.from_matrix_to_params(h_noisy_denormalized.unsqueeze(0) * AtomicUnits.Eh)
    h_noisy_map_json = tensor_dict_to_list(h_noisy_map)
    with open(os.path.join(sample_dir, 'h_noisy_map.json'), 'w') as f:
        json.dump(h_noisy_map_json, f, indent=4)

    test_noisy_hamiltonian = TorchHamiltonian.from_2channel_tensor_with_params(h_noisy_denormalized, dit_encoder_config['input_hamiltonian_params'])
    eigvals_occ_path = os.path.join(sample_dir, eigvals_plot_name.format(f'occ_noisy'))
    plot_eigvals(test_noisy_hamiltonian, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)
    eigvals_eh_path = os.path.join(sample_dir, eigvals_plot_name.format(f'eh_noisy'))
    plot_eigvals(test_noisy_hamiltonian, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)

    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_noisy_denormalized[0].detach().cpu().numpy(), test_matrix_path.format('noisy_real'), vmin=-vscale, vmax=vscale, title=f'M: {h_noisy_label:.2f}')
    plot_matrix(h_noisy_denormalized[1].detach().cpu().numpy(), test_matrix_path.format('noisy_imag'), vmin=-vscale, vmax=vscale, title=f'M: {h_noisy_label:.2f}')

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

    h_predicted_map = hamiltonian_converter.from_matrix_to_params(h_predicted_denorm.unsqueeze(0) * AtomicUnits.Eh)
    h_predicted_map_json = tensor_dict_to_list(h_predicted_map)
    with open(os.path.join(sample_dir, 'h_predicted_map.json'), 'w') as f:
        json.dump(h_predicted_map_json, f, indent=4)

    h_complex_tensor = torch.complex(h_predicted_denorm[0], h_predicted_denorm[1])
    h_predicted_label = majoranization(h_complex_tensor.detach().cpu().numpy(), n_dots)

    eigvals_dit_path = os.path.join(sample_dir, eigvals_plot_name.format(f'GEN_rec'))

    ham_rec = TorchHamiltonian.from_2channel_tensor_with_params(h_predicted_denorm, dit_encoder_config['input_hamiltonian_params'])
    eigvals_occ_path = os.path.join(sample_dir, eigvals_plot_name.format(f'occ_GEN_rec'))
    plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)
    eigvals_eh_path = os.path.join(sample_dir, eigvals_plot_name.format(f'eh_GEN_rec'))
    plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)
    
    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_predicted_denorm[0].detach().cpu().numpy(), test_matrix_path.format('GEN_improved_real'), vmin=-vscale, vmax=vscale, title=f'M: {h_predicted_label:.2f}')
    plot_matrix(h_predicted_denorm[1].detach().cpu().numpy(), test_matrix_path.format('GEN_improved_imag'), vmin=-vscale, vmax=vscale, title=f'M: {h_predicted_label:.2f}')

    for i, cmap_config in enumerate(conductance_config_hq['cmap_list']):
        noisy_conductance_path = os.path.join(sample_dir, 'noisy_conductance')
        os.makedirs(noisy_conductance_path, exist_ok=True)
        cmap_name = list(cmap_config.keys())[0]
        x_name = 'b' if cmap_name == 'cmap2' else 'mu'
        x_label = 'B' if cmap_name == 'cmap2' else '$\mu$'

        x_tick_range = np.array(cmap_config[cmap_name][f'{x_name}_range']) * AtomicUnits.Eh
        y_tick_range = np.array(cmap_config[cmap_name]['ef_range']) * AtomicUnits.Eh
        i_val = cmap_config[cmap_name]['i']
        j_val = cmap_config[cmap_name]['j']
        if cmap_name == 'cmap0':
            filename_suffix = f'_dot{cmap_config[cmap_name]["site_index"]}'
        else:
            filename_suffix = ''

        # h_noisy_conductance = noisy_cmap_denormalize(h_noisy_conductance)
        plot_conductance_map(
            h_noisy_conductance_hq[i].detach().cpu().numpy(),
            os.path.join(noisy_conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{filename_suffix}.png'),
            xtick_range=x_tick_range,
            ytick_range=y_tick_range,
            xlabel=x_label,
            ylabel="$E_F$",
            title=f'Noise: {real_noise_amplitude.item():.2f}, M: {h_noisy_label:.2f}'
        )

    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        ref_conductance_path = os.path.join(sample_dir, 'ref_conductance')
        os.makedirs(ref_conductance_path, exist_ok=True)
        cmap_name = list(cmap_config.keys())[0]
        x_name = 'b' if cmap_name == 'cmap2' else 'mu'
        x_label = 'B' if cmap_name == 'cmap2' else '$\mu$'

        x_tick_range = np.array(cmap_config[cmap_name][f'{x_name}_range']) * AtomicUnits.Eh
        y_tick_range = np.array(cmap_config[cmap_name]['ef_range']) * AtomicUnits.Eh
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
            xlabel=x_label,
            ylabel="$E_F$",
            title=f'M: {h_label:.2f}'
        )

    mapped_h_predicted = torch.complex(h_predicted_denorm[0], h_predicted_denorm[1])
    predicted_cmap = generate_conductance_tensor(mapped_h_predicted, conductance_config_hq)
    
    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        predicted_conductance_path = os.path.join(sample_dir, 'GEN_improved_conductance')
        os.makedirs(predicted_conductance_path, exist_ok=True)
        cmap_name = list(cmap_config.keys())[0]
        x_name = 'b' if cmap_name == 'cmap2' else 'mu'
        x_label = 'B' if cmap_name == 'cmap2' else '$\mu$'

        x_tick_range = np.array(cmap_config[cmap_name][f'{x_name}_range']) * AtomicUnits.Eh
        y_tick_range = np.array(cmap_config[cmap_name]['ef_range']) * AtomicUnits.Eh
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
            xlabel=x_label,
            ylabel="$E_F$",
            title=f'M: {h_predicted_label:.2f}'
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
    h_rec_label = majoranization(h_complex_tensor.detach().cpu().numpy(), n_dots)
    
    eigvals_dit_path = os.path.join(sample_dir, eigvals_plot_name.format(f'AE_rec'))

    ham_rec = TorchHamiltonian.from_2channel_tensor_with_params(h_rec_denorm, dit_encoder_config['input_hamiltonian_params'])
    eigvals_occ_path = os.path.join(sample_dir, eigvals_plot_name.format(f'occ_AE_rec'))
    plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)
    eigvals_eh_path = os.path.join(sample_dir, eigvals_plot_name.format(f'eh_AE_rec'))
    plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)
        
    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_rec_denorm[0].detach().cpu().numpy(), test_matrix_path.format('AE_reconstructed_real'), vmin=-vscale, vmax=vscale, title=f'M: {h_rec_label:.2f}')
    plot_matrix(h_rec_denorm[1].detach().cpu().numpy(), test_matrix_path.format('AE_reconstructed_imag'), vmin=-vscale, vmax=vscale, title=f'M: {h_rec_label:.2f}')

    mapped_h_predicted = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
    cmap_rec = generate_conductance_tensor(mapped_h_predicted, conductance_config)
    
    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        predicted_conductance_path = os.path.join(sample_dir, 'AE_reconstructed_conductance')
        os.makedirs(predicted_conductance_path, exist_ok=True)
        cmap_name = list(cmap_config.keys())[0]
        x_name = 'b' if cmap_name == 'cmap2' else 'mu'
        x_label = 'B' if cmap_name == 'cmap2' else '$\mu$'

        x_tick_range = np.array(cmap_config[cmap_name][f'{x_name}_range']) * AtomicUnits.Eh
        y_tick_range = np.array(cmap_config[cmap_name]['ef_range']) * AtomicUnits.Eh
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
            xlabel=x_label,
            ylabel="$E_F$",
            title=f'M: {h_rec_label:.2f}'
        )

    # Noisy 
    latent_h = model_encoder(h_perturbed, mock_noise_amplitude, None)
    reconstructed_h = model_decoder(latent_h, mock_noise_amplitude, None)

    h_rec_denorm = h_denormalize(reconstructed_h)[0]
    h_complex_tensor = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
    h_rec_noisy_label = majoranization(h_complex_tensor.detach().cpu().numpy(), n_dots)
    
    eigvals_dit_path = os.path.join(sample_dir, eigvals_plot_name.format(f'AE_rec_noisy'))

    ham_rec = TorchHamiltonian.from_2channel_tensor_with_params(h_rec_denorm, dit_encoder_config['input_hamiltonian_params'])
    eigvals_occ_path = os.path.join(sample_dir, eigvals_plot_name.format(f'occ_AE_rec_noisy'))
    plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)
    eigvals_eh_path = os.path.join(sample_dir, eigvals_plot_name.format(f'eh_AE_rec_noisy'))
    plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)

    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_rec_denorm[0].detach().cpu().numpy(), test_matrix_path.format('AE_reconstructed_noisy_real'), vmin=-vscale, vmax=vscale, title=f'M: {h_rec_noisy_label:.2f}')
    plot_matrix(h_rec_denorm[1].detach().cpu().numpy(), test_matrix_path.format('AE_reconstructed_noisy_imag'), vmin=-vscale, vmax=vscale, title=f'M: {h_rec_noisy_label:.2f}')

    mapped_h_predicted = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
    cmap_rec = generate_conductance_tensor(mapped_h_predicted, conductance_config)
    
    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        predicted_conductance_path = os.path.join(sample_dir, 'AE_reconstructed_noisy_conductance')
        os.makedirs(predicted_conductance_path, exist_ok=True)
        cmap_name = list(cmap_config.keys())[0]
        x_name = 'b' if cmap_name == 'cmap2' else 'mu'
        x_label = 'B' if cmap_name == 'cmap2' else '$\mu$'

        x_tick_range = np.array(cmap_config[cmap_name][f'{x_name}_range']) * AtomicUnits.Eh
        y_tick_range = np.array(cmap_config[cmap_name]['ef_range']) * AtomicUnits.Eh
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
            xlabel=x_label,
            ylabel="$E_F$",
            title=f'M: {h_rec_noisy_label:.2f}'
        )

    # AE from GEN
    latent_h = model_encoder(h_predicted, mock_noise_amplitude, None)
    reconstructed_h = model_decoder(latent_h, mock_noise_amplitude, None)

    h_rec_denorm = h_denormalize(reconstructed_h)[0]
    h_complex_tensor = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
    h_rec_imp_label = majoranization(h_complex_tensor.detach().cpu().numpy(), n_dots)
    
    eigvals_dit_path = os.path.join(sample_dir, eigvals_plot_name.format(f'AE_GEN_rec'))

    ham_rec = TorchHamiltonian.from_2channel_tensor_with_params(h_rec_denorm, dit_encoder_config['input_hamiltonian_params'])
    eigvals_occ_path = os.path.join(sample_dir, eigvals_plot_name.format(f'occ_AE_GEN_rec'))
    plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_occ_path, color='occupations', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)
    eigvals_eh_path = os.path.join(sample_dir, eigvals_plot_name.format(f'eh_AE_GEN_rec'))
    plot_eigvals(ham_rec, xaxis=x_axis, xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', xnorm=xnorm, ynorm=ynorm, ylim=ylim, majoranization=True)

    test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
    plot_matrix(h_rec_denorm[0].detach().cpu().numpy(), test_matrix_path.format('AE_GEN_reconstructed_real'), vmin=-vscale, vmax=vscale, title=f'M: {h_rec_imp_label:.2f}')
    plot_matrix(h_rec_denorm[1].detach().cpu().numpy(), test_matrix_path.format('AE_GEN_reconstructed_imag'), vmin=-vscale, vmax=vscale, title=f'M: {h_rec_imp_label:.2f}')

    mapped_h_predicted = torch.complex(h_rec_denorm[0], h_rec_denorm[1])
    cmap_rec = generate_conductance_tensor(mapped_h_predicted, conductance_config)
    
    for i, cmap_config in enumerate(conductance_config['cmap_list']):
        predicted_conductance_path = os.path.join(sample_dir, 'AE_GEN_reconstructed_conductance')
        os.makedirs(predicted_conductance_path, exist_ok=True)
        cmap_name = list(cmap_config.keys())[0]
        x_name = 'b' if cmap_name == 'cmap2' else 'mu'
        x_label = 'B' if cmap_name == 'cmap2' else '$\mu$'

        x_tick_range = np.array(cmap_config[cmap_name][f'{x_name}_range']) * AtomicUnits.Eh
        y_tick_range = np.array(cmap_config[cmap_name]['ef_range']) * AtomicUnits.Eh
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
            xlabel=x_label,
            ylabel="$E_F$",
            title=f'M: {h_rec_imp_label:.2f}'
        )     
