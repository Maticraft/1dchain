import json
import os
import pickle

from copy import deepcopy

from matplotlib import pyplot as plt
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
from src.plots import plot_convergence, plot_matrix
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
# epoch = 170  # 260
epoch = 130

# Reference eigvals plot params
tests_sub_dir = f'noise1_reconstruction_tests_ep_{epoch}'
x_axis = 'potential'
x_values = np.linspace(-.6/AtomicUnits.Eh, .6/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-0.75, 0.75)
vscale = 1/AtomicUnits.Eh
n_dots = 3

# Device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

# Model name
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-GEN-KL-latent-loss_2DiT-AE-patch4-1_improve-selected-params_gradually-increasing-noise10_4maps50x50embed_norm-c2c_lr1e-4-decreasing'

# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg1_2DiT-AE-patch4-1_improve-selected-params_majorana-destructing-noise1_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg1_2DiT-AE-patch4-1_improve-selected-params_majorana-destructing-noise1-extra-params_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'

# New model
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-b-const-params_majorana-destructing-noise1-extra-params_relative-1b-3mu-maps50x50_norm-c2c_lr1e-4-decreasing'


# New model all maps
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-b-const-params_majorana-destructing-noise1-extra-params_relative-4b-12mu-maps50x50_norm-c2c_lr1e-4-decreasing'
model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-t-l-params_majorana-destructing-noise1-extra-params_relative-4b-12mu-maps50x50_norm-c2c_lr1e-4-decreasing'


# Paper model
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
    'reg_strength': 1.,
    'site_independent_noise': True,  # If False, the noise is applied to all sites equally
}


# Set params
# sample_name = 'noisy_params_mu_0_75_d_0_7'
sample_name = 'noisy_params_l0_1_95_l1_1_3'
# sample_name = 'noisy_params_t0_0_06_t1_0_06'
# sample_name = 'noisy_params_l0_3_14_l1_2_826'

# sample_name = 'default_params'
defaults = DefaultParameters()
# defaults.t_default = 2./AtomicUnits.Eh
# defaults.l_default = 1.5
# defaults.mu_default = 0.75/AtomicUnits.Eh
# defaults.b_default = 0.27/AtomicUnits.Eh
# defaults.d_default = 0.7/AtomicUnits.Eh

h_params = QuantumDotsHamiltonianParameters(3, 1, defaults)

# h_params.b[2] = 1.75/AtomicUnits.Eh
# h_params.d[2] = 1./AtomicUnits.Eh
# h_params.mu[0] = 1.15/AtomicUnits.Eh
# h_params.mu[1] = 1.2/AtomicUnits.Eh
# h_params.t[0] = 0.06/AtomicUnits.Eh
# h_params.t[1] = 0.06/AtomicUnits.Eh
h_params.l[0] = 1.95
h_params.l[1] = 1.3
# h_params.l[0] = 3.14
# h_params.l[1] = 2.826
# h_params.d[0] = 0.25/AtomicUnits.Eh
# h_params.d[1] = 1.25/AtomicUnits.Eh

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

num_parameters_c2h = sum(p.numel() for p in model_c2h.parameters() if p.requires_grad)
print(f'Number of trainable parameters in model_c2h: {num_parameters_c2h}')



h_mean, h_std = normalization_params[0]
h_normalize = Normalize(mean=h_mean, std=h_std)
h_denormalize = Denormalize(mean=h_mean, std=h_std)

cmap_mean, cmap_std = normalization_params[1]
cmap_normalize = Normalize(mean=cmap_mean, std=cmap_std)
cmap_denormalize = Denormalize(mean=cmap_mean, std=cmap_std)

noisy_cmap_mean, noisy_cmap_std = normalization_params[3]
noisy_cmap_denormalize = Denormalize(mean=noisy_cmap_mean, std=noisy_cmap_std)


sample_dir = os.path.join(tests_sub_path, sample_name)
os.makedirs(sample_dir, exist_ok=True)

hamiltonian = QuantumDotsHamiltonian(h_params)
h_tensor = hamiltonian.get_hamiltonian_tensor()
h_tensor_complex = torch.complex(h_tensor[0], h_tensor[1]).to(device)

eigvals_test_path = os.path.join(sample_dir, eigvals_plot_name.format(f'reference'))
    # h_torch_normalized = test_sample[0][0].unsqueeze(0).to(device)
    # h_torch_denormalized = h_denormalize(h_torch_normalized)[0]
    # h_complex_tensor = torch.complex(h_torch_denormalized[0], h_torch_denormalized[1])
h_label = majoranization(h_tensor_complex.detach().cpu().numpy(), n_dots)

h_map = hamiltonian_converter.from_matrix_to_params(h_tensor.unsqueeze(0) * AtomicUnits.Eh)
h_map_json = tensor_dict_to_list(h_map)
with open(os.path.join(sample_dir, 'h_original_map.json'), 'w') as f:
    json.dump(h_map_json, f, indent=4)

# test_hamiltonian = TorchHamiltonian.from_2channel_tensor_with_params(h_tensor, dit_encoder_config['input_hamiltonian_params'])


conductance = generate_conductance_tensor(h_tensor_complex, conductance_config)
conductance_normalized = cmap_normalize(conductance).unsqueeze(0)

noise_amplitude = 0.1
t = torch.tensor(noise_amplitude, device=device).view(1, 1)
improve_map = model_c2h.forward_to_params(conductance_normalized, t, None)

h_tensor_norm = h_normalize(h_tensor.unsqueeze(0)).to(device)
h_map = hamiltonian_converter.from_matrix_to_params(h_tensor_norm)
improved_map = deep_update(h_map, improve_map, detach=False)


h_predicted = hamiltonian_converter.from_params_to_matrix(improved_map)
h_predicted_denorm = h_denormalize(h_predicted)[0]
test_hamiltonian = TorchHamiltonian.from_2channel_tensor_with_params(h_predicted_denorm, dit_encoder_config['input_hamiltonian_params'])

h_predicted_map = hamiltonian_converter.from_matrix_to_params(h_predicted_denorm.unsqueeze(0) * AtomicUnits.Eh)
h_predicted_map_json = tensor_dict_to_list(h_predicted_map)
with open(os.path.join(sample_dir, 'h_denoised_map.json'), 'w') as f:
    json.dump(h_predicted_map_json, f, indent=4)


plt.rcParams.update({'font.size': 20})

fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

eigvals_occ_path = os.path.join(sample_dir, eigvals_plot_name.format(f'occ_original'))
plot1 = plot_eigvals(hamiltonian, xaxis='mu', xparams=x_values, filename=eigvals_occ_path, color='occupations', ylim=ylim, majoranization=True, xnorm=xnorm, ynorm=ynorm, ax=axs[0], right_label=False)
eigvals_eh_path = os.path.join(sample_dir, eigvals_plot_name.format(f'eh_original'))
plot2 = plot_eigvals(hamiltonian, xaxis='mu', xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', ylim=ylim, majoranization=True, xnorm=xnorm, ynorm=ynorm, ax=axs[1], left_label=False)

plt.subplots_adjust(wspace=0.02)

cbar1 = fig.colorbar(plot1, ax=axs[0], orientation='horizontal', fraction=0.035, pad=0.18)
cbar1.set_label('edge occupations')
vmin1, vmax1 = plot1.get_clim()
center1 = (vmin1 + vmax1) / 2
cbar1.set_ticks([vmin1, center1, vmax1])
cbar1.set_ticklabels([f'{vmin1:.1f}', f'{center1:.1f}', f'{vmax1:.1f}'])

cbar2 = fig.colorbar(plot2, ax=axs[1], orientation='horizontal', fraction=0.035, pad=0.18)
cbar2.set_label('electron-hole symmetry')
vmin2, vmax2 = plot2.get_clim()
center2 = (vmin2 + vmax2) / 2
cbar2.set_ticks([vmin2, center2, vmax2])
cbar2.set_ticklabels([f'{vmin2:.1f}', f'{center2:.1f}', f'{vmax2:.1f}'])

plt.savefig(os.path.join(sample_dir, eigvals_plot_name.format(f'original')), bbox_inches='tight', dpi=300)
plt.close(fig)


plt.rcParams.update({'font.size': 20})

fig, axs = plt.subplots(2, 2, figsize=(10, 12), sharey=True)

eigvals_occ_path = os.path.join(sample_dir, eigvals_plot_name.format(f'occ_original'))
plot1 = plot_eigvals(hamiltonian, xaxis='mu', xparams=x_values, filename=eigvals_occ_path, color='occupations', ylim=ylim, majoranization=True, xnorm=xnorm, ynorm=ynorm, ax=axs[0, 0], right_label=False, x_vline=defaults.mu_default * AtomicUnits.Eh)
eigvals_eh_path = os.path.join(sample_dir, eigvals_plot_name.format(f'eh_original'))
plot2 = plot_eigvals(hamiltonian, xaxis='mu', xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', ylim=ylim, majoranization=True, xnorm=xnorm, ynorm=ynorm, ax=axs[0, 1], left_label=False, x_vline=defaults.mu_default * AtomicUnits.Eh)

# plt.subplots_adjust(wspace=0.02)

# cbar1 = fig.colorbar(plot1, ax=axs[0], orientation='horizontal', fraction=0.035, pad=0.18)
# cbar1.set_label('edge occupations')
# vmin1, vmax1 = plot1.get_clim()
# center1 = (vmin1 + vmax1) / 2
# cbar1.set_ticks([vmin1, center1, vmax1])
# cbar1.set_ticklabels([f'{vmin1:.1f}', f'{center1:.1f}', f'{vmax1:.1f}'])


# cbar2 = fig.colorbar(plot2, ax=axs[1], orientation='horizontal', fraction=0.035, pad=0.18)
# cbar2.set_label('electron-hole symmetry')
# vmin2, vmax2 = plot2.get_clim()
# center2 = (vmin2 + vmax2) / 2
# cbar2.set_ticks([vmin2, center2, vmax2])
# cbar2.set_ticklabels([f'{vmin2:.1f}', f'{center2:.1f}', f'{vmax2:.1f}'])

# plt.savefig(os.path.join(sample_dir, eigvals_plot_name.format(f'original')), bbox_inches='tight', dpi=300)
# plt.close(fig)


# plt.rcParams.update({'font.size': 20})
# fig, axs = plt.subplots(1, 2, figsize=(10, 6), sharey=True)

eigvals_occ_path = os.path.join(sample_dir, eigvals_plot_name.format(f'occ_denoised'))
plot1 = plot_eigvals(test_hamiltonian, xaxis='potential', xparams=x_values, filename=eigvals_occ_path, color='occupations', ylim=ylim, majoranization=True, xnorm=xnorm, ynorm=ynorm, ax=axs[1, 0], right_label=False, x_vline=defaults.mu_default * AtomicUnits.Eh)
eigvals_eh_path = os.path.join(sample_dir, eigvals_plot_name.format(f'eh_denoised'))
plot2 = plot_eigvals(test_hamiltonian, xaxis='potential', xparams=x_values, filename=eigvals_eh_path, color='electron-hole-diff', ylim=ylim, majoranization=True, xnorm=xnorm, ynorm=ynorm, ax=axs[1, 1], left_label=False, x_vline=defaults.mu_default * AtomicUnits.Eh)

plt.subplots_adjust(wspace=0.02, hspace=0.25)

cbar1 = fig.colorbar(plot1, ax=axs[:, 0], orientation='horizontal', fraction=0.017, pad=0.1)
cbar1.set_label('edge occupations')
vmin1, vmax1 = plot1.get_clim()
center1 = (vmin1 + vmax1) / 2
cbar1.set_ticks([vmin1, center1, vmax1])
cbar1.set_ticklabels([f'{vmin1:.1f}', f'{center1:.1f}', f'{vmax1:.1f}'])


cbar2 = fig.colorbar(plot2, ax=axs[:, 1], orientation='horizontal', fraction=0.017, pad=0.1)
cbar2.set_label('electron-hole symmetry')
vmin2, vmax2 = plot2.get_clim()
center2 = (vmin2 + vmax2) / 2
cbar2.set_ticks([vmin2, center2, vmax2])
cbar2.set_ticklabels([f'{vmin2:.1f}', f'{center2:.1f}', f'{vmax2:.1f}'])

# Add a) label for axs[0] and b) label fors axs[1] on the left side of the subplots
axs[0, 0].text(-0.15, 1.05, 'a)', transform=axs[0, 0].transAxes, fontsize=25, fontweight='bold', va='top', ha='right')
axs[1, 0].text(-0.15, 1.05, 'b)', transform=axs[1, 0].transAxes, fontsize=25, fontweight='bold', va='top', ha='right')

plt.savefig(os.path.join(sample_dir, eigvals_plot_name.format(f'both')), bbox_inches='tight', dpi=300)
plt.close(fig)


test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
plot_matrix(h_tensor[0].detach().cpu().numpy(), test_matrix_path.format('original_real'), vmin=-vscale, vmax=vscale, title=f'M: {h_label:.2f}')
plot_matrix(h_tensor[1].detach().cpu().numpy(), test_matrix_path.format('original_imag'), vmin=-vscale, vmax=vscale, title=f'M: {h_label:.2f}')


h_conductance_hq = generate_conductance_tensor(h_tensor_complex, conductance_config_hq)

plt.rcParams.update({'font.size': 30})
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharey=True)

index_map = {
    1: 'L',
    2: 'C',
    3: 'R',
}

for i, cmap_config in enumerate(conductance_config_hq['cmap_list'][:4]):
    conductance_path = os.path.join(sample_dir, 'original_conductance')
    os.makedirs(conductance_path, exist_ok=True)
    cmap_name = list(cmap_config.keys())[0]
    x_name = 'b' if cmap_name == 'cmap2' else 'mu'
    
    if cmap_name == 'cmap2':
        x_label = '$\Delta B_Z$'
    else:
        index = cmap_config[cmap_name]['site_index'] + 1
        index_label = index_map.get(index, str(index))
        x_label = r'$\Delta \mu_{}$'.format(index_label)

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
        h_conductance_hq[i].detach().cpu().numpy(),
        os.path.join(conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{filename_suffix}.png'),
        xtick_range=x_tick_range,
        ytick_range=y_tick_range,
        xlabel=f"{x_label} (meV)",
        ylabel="$E_F$ (meV)" if i//2 == 0 else '',
        title='', # r'$\mathcal{M}$: ' + f'{h_label:.2f}'
        ax=axs[i%2, i//2],
        right_xtick=i//2==1
    )

# If font = 30
plt.subplots_adjust(wspace=-0.4, hspace=0.4)  # Reduce horizontal and vertical spacing

# If font = 20
# plt.subplots_adjust(wspace=-0.37, hspace=0.3)  # Reduce horizontal and vertical spacing


cbar = fig.colorbar(axs[0,0].images[0], ax=axs, orientation='vertical', fraction=0.035, pad=0.05)
cbar.ax.set_title('$G_{LL}$', pad=20)
# plt.tight_layout()
plt.savefig(os.path.join(conductance_path, 'all_conductance_maps.png'), bbox_inches='tight', dpi=300)
plt.close(fig)

# ----------------------------------------------------
# Denoised plots

h_complex_tensor = torch.complex(h_predicted_denorm[0], h_predicted_denorm[1]).detach().cpu()
h_predicted_label = majoranization(h_complex_tensor.numpy(), n_dots)

test_matrix_path = os.path.join(sample_dir, hamiltonian_plot_name)
plot_matrix(h_tensor[0].detach().cpu().numpy(), test_matrix_path.format('denoised_real'), vmin=-vscale, vmax=vscale, title=f'M: {h_predicted_label:.2f}')
plot_matrix(h_tensor[1].detach().cpu().numpy(), test_matrix_path.format('denoised_imag'), vmin=-vscale, vmax=vscale, title=f'M: {h_predicted_label:.2f}')

h_conductance_hq = generate_conductance_tensor(h_complex_tensor, conductance_config_hq)

plt.rcParams.update({'font.size': 30})
fig, axs = plt.subplots(2, 2, figsize=(10, 8), sharey=True)

for i, cmap_config in enumerate(conductance_config_hq['cmap_list'][:4]):
    conductance_path = os.path.join(sample_dir, 'denoised_conductance')
    os.makedirs(conductance_path, exist_ok=True)
    cmap_name = list(cmap_config.keys())[0]
    x_name = 'b' if cmap_name == 'cmap2' else 'mu'
    if cmap_name == 'cmap2':
        x_label = '$\Delta B_Z$'
    else:
        index = cmap_config[cmap_name]['site_index'] + 1
        index_label = index_map.get(index, str(index))
        x_label = r'$\Delta \mu_{}$'.format(index_label)

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
        h_conductance_hq[i].detach().cpu().numpy(),
        os.path.join(conductance_path, f'{cmap_name}_i{i_val}_j{j_val}{filename_suffix}.png'),
        xtick_range=x_tick_range,
        ytick_range=y_tick_range,
        xlabel=f"{x_label} (meV)",
        ylabel="$E_F$ (meV)" if i//2 == 0 else '',
        title='', # r'$\mathcal{M}$: ' + f'{h_predicted_label:.2f}'
        ax=axs[i%2, i//2],
        right_xtick=i//2==1
    )

# If font = 30
plt.subplots_adjust(wspace=-0.4, hspace=0.4)  # Reduce horizontal and vertical spacing

# If font = 20
# plt.subplots_adjust(wspace=-0.37, hspace=0.3)  # Reduce horizontal and vertical spacing

cbar = fig.colorbar(axs[0,0].images[0], ax=axs, orientation='vertical', fraction=0.035, pad=0.05)
cbar.ax.set_title("$G_{LL}'$", pad=25)
# plt.tight_layout()
plt.savefig(os.path.join(conductance_path, 'all_conductance_maps.png'), bbox_inches='tight', dpi=300)
plt.close(fig)
