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
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonian, QuantumDotsHamiltonianParameters
from src.hamiltonian.utils import majoranization, plot_eigvals, plot_eigvals_levels
from src.models.denoiser import train_denoising_conductance_DiTGEN_AE, test_denoising_conductance_DiTGEN_AE
from src.models.files import save_params, save_model, save_data_list, load_params, load_model
from src.models.diffusion_transformer import DiT
from src.models.utils import deep_update, tensor_dict_to_list, weighted_update
from src.plots import plot_convergence, plot_matrix, plot_majoranization_denoising_map, plot_majoranization_denoising_map_from_file
from src.hamiltonian.utils import plot_eigvals_levels
from src.hamiltonian.torch_hamiltonian import TorchHamiltonian

# Paths
data_path = './data/quantum_dots/simple_fixed_t_3dots1level_majoranization0_0_mzm_gap0_25'
# normalization_params_path = f'{data_path}/std_rep_normalization_params_4b-maps.pkl'
# normalization_params_path = f'{data_path}/std_rep_normalization_params_1b-3mu-maps.pkl'
normalization_params_path = f'{data_path}/std_rep_normalization_params_4b-12mu-maps.pkl'

save_dir = './conductance/quantum_dots/simple_fixed_t_3dots1level_majoranization0_0_mzm_gap0_25'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'
distribution_dir_name = 'tests_majoranas_latent_ep_{}'
eigvals_plot_name = 'eigvals_spectre_{}.png'
hamiltonian_plot_name = 'hamiltonian_{}.png'


num_samples = 10
# epoch = 160
# epoch = 170
# epoch = 260

# epoch = 170
epoch = 130

# Reference eigvals plot params
tests_sub_dir = f'noise1_reconstruction_tests_ep_{epoch}'
x_axis = 'potential'
x_values = np.linspace(-.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-2., 2.)
vscale = 1/AtomicUnits.Eh
n_dots = 3

plot_legend = False
margin = 0.

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model name
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-GEN-KL-latent-loss_2DiT-AE-patch4-1_improve-selected-params_gradually-increasing-noise10_4maps50x50embed_norm-c2c_lr1e-4-decreasing'

# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg1_2DiT-AE-patch4-1_improve-selected-params_majorana-destructing-noise1_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg1_2DiT-AE-patch4-1_improve-selected-params_majorana-destructing-noise1-extra-params_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'

# New model
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-b-params_majorana-destructing-noise1-extra-params_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-b-const-params_majorana-destructing-noise1-extra-params_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'

# New model all maps
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-b-const-params_majorana-destructing-noise1-extra-params_relative-4b-12mu-maps50x50_norm-c2c_lr1e-4-decreasing'
model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-t-l-params_majorana-destructing-noise1-extra-params_relative-4b-12mu-maps50x50_norm-c2c_lr1e-4-decreasing'

# paper model
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-params_majorana-destructing-noise1-extra-params_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'

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


dot_index_x = None  # Index of the dot to plot
dot_index_y = None  # Index of the dot to plot
if dot_index_x is None:
    suffix = ''
else:
    suffix = f'_x-dot_{dot_index_x}'

if dot_index_y is None:
    suffix += ''
else:
    suffix += f'_y-dot_{dot_index_y}'

x_param_name = 'mu'
y_param_name = 't'

defaults = DefaultParameters()

x_default = getattr(defaults, f'{x_param_name}_default')
x_dataset_range = getattr(defaults, f'{x_param_name}_range')

y_default = getattr(defaults, f'{y_param_name}_default')
y_dataset_range = getattr(defaults, f'{y_param_name}_range')

if dot_index_x is None:
    setattr(defaults, f'{x_param_name}_default', 0.)

if dot_index_y is None:
    setattr(defaults, f'{y_param_name}_default', 0.)

h_params = QuantumDotsHamiltonianParameters(3, 1, defaults)

if dot_index_x is not None:
    x_array = getattr(h_params, f'{x_param_name}')
    x_array[dot_index_x] = 0.
    setattr(h_params, f'{x_param_name}', x_array)

if dot_index_y is not None:
    y_array = getattr(h_params, f'{y_param_name}')
    y_array[dot_index_y] = 0.
    setattr(h_params, f'{y_param_name}', y_array)


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
#         'cmap0': {
#             'i':0,
#             'j':0,
#             'gamma': 0.1,
#             'mu_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
#             'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
#             'mu_num': 50,
#             'ef_num': 50,
#             'site_index': 1,
#             'with_embedding': False,
#         },
#     },
#     {
#         'cmap0': {
#             'i':0,
#             'j':0,
#             'gamma': 0.1,
#             'mu_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
#             'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
#             'mu_num': 50,
#             'ef_num': 50,
#             'site_index': 0,
#             'with_embedding': False,
#         },
#     },
#     {
#         'cmap0': {
#             'i':0,
#             'j':0,
#             'gamma': 0.1,
#             'mu_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
#             'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
#             'mu_num': 50,
#             'ef_num': 50,
#             'site_index': 2,
#             'with_embedding': False
#         },
#     },
#     ],
# }


cmap_lists = [
    [
        {
            'cmap2': {
                'i':i,
                'j':j,
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
                'i':i,
                'j':j,
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
                'i':i,
                'j':j,
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
                'i':i,
                'j':j,
                'gamma': 0.1,
                'mu_range': (-0.5/AtomicUnits.Eh, 0.5/AtomicUnits.Eh),
                'ef_range': (-1./AtomicUnits.Eh, 1./AtomicUnits.Eh),
                'mu_num': 50,
                'ef_num': 50,
                'site_index': 2,
                'with_embedding': False
            },
        },
    ] for i in range(2) for j in range(2)
]

conductance_config = {
    'cmap_list': [cmap for sublist in cmap_lists for cmap in sublist],
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


h_mean, h_std = normalization_params[0]
h_normalize = Normalize(mean=h_mean, std=h_std)
h_denormalize = Denormalize(mean=h_mean, std=h_std)

cmap_mean, cmap_std = normalization_params[1]
cmap_normalize = Normalize(mean=cmap_mean, std=cmap_std)
cmap_denormalize = Denormalize(mean=cmap_mean, std=cmap_std)

noisy_cmap_mean, noisy_cmap_std = normalization_params[3]
noisy_cmap_denormalize = Denormalize(mean=noisy_cmap_mean, std=noisy_cmap_std)


l = params['max_noise_amplitude']

filename = f'majoranization_denoising_map_10it_dec1_no_weights_{x_param_name}_{y_param_name}{suffix}'
save_log = os.path.join(tests_sub_path, f'{filename}_log.txt')
save_plot_path = os.path.join(tests_sub_path, f'{filename}.png')

try:
    x_param_name_with_index = f'{x_param_name}_{dot_index_x}' if dot_index_x is not None else x_param_name
    y_param_name_with_index = f'{y_param_name}_{dot_index_y}' if dot_index_y is not None else y_param_name
    plot_majoranization_denoising_map_from_file(
        save_log,
        save_plot_path,
        x_param=x_param_name_with_index,
        y_param=y_param_name_with_index,
        default_x_value=x_default,
        default_y_value=y_default,
        denormalize_x=True if x_param_name != 'l' else False,
        denormalize_y=True if y_param_name != 'l' else False,
        x_range=(0., 1.5/AtomicUnits.Eh) if x_param_name != 'l' else (0., np.pi),
        y_range=(0., 1.5/AtomicUnits.Eh) if y_param_name != 'l' else (0., np.pi),
        x_dataset_range=x_dataset_range if x_param_name != 'l' else None,
        y_dataset_range=y_dataset_range if y_param_name != 'l' else None,
        normalize_majoranization=False,
        legend=plot_legend,
        pad_inches=margin,
    )
except FileNotFoundError:
    with torch.no_grad():
        plot_majoranization_denoising_map(
            denoiser=model_c2h,
            hamiltonian_class=QuantumDotsHamiltonian,
            start_params=h_params.to_dict(),
            x_param=x_param_name,
            x_range=(0./AtomicUnits.Eh, 1.5/AtomicUnits.Eh) if x_param_name != 'l' else (0., np.pi),
            y_param=y_param_name,
            y_range=(0., 1.5/AtomicUnits.Eh) if y_param_name != 'l' else (0., np.pi),
            cmap_config=conductance_config,
            save_path=save_plot_path,
            cmap_normalize=cmap_normalize,
            h_normalize=h_normalize,
            h_denormalize=h_denormalize,
            hamiltonian_converter=hamiltonian_converter,
            resolution=50,
            n_dots=n_dots,
            noise_amplitude=1.,
            save_log=save_log,
            dot_index_x=dot_index_x,
            dot_index_y=dot_index_y,
            iterations=10
        )
