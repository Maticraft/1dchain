import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.quantum_dots_chain import DefaultParameters, QuantumDotsHamiltonianParameters
from src.plots import plot_majoranization_denoising_map_from_file


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
# epoch = 160
# epoch = 170
# epoch = 260
epoch = 170
# epoch = 130

# Reference eigvals plot params
tests_sub_dir = f'noise1_reconstruction_tests_ep_{epoch}'
x_axis = 'potential'
x_values = np.linspace(-.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-2., 2.)
vscale = 1/AtomicUnits.Eh
n_dots = 3

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
model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-b-const-params_majorana-destructing-noise1-extra-params_relative-4b-12mu-maps50x50_norm-c2c_lr1e-4-decreasing'
# model_name = 'Hamiltionian-QDH-1lvl-no-interlevel_DiT-mu-fixed_min-majoranization-loss-eh-with-reg0-1_2DiT-AE-patch4-1_improve-selected-mu-t-l-params_majorana-destructing-noise1-extra-params_relative-4b-12mu-maps50x50_norm-c2c_lr1e-4-decreasing'


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


dot_index_x = 0  # Index of the dot to plot
dot_index_y = 1  # Index of the dot to plot
if dot_index_x is None:
    suffix = ''
else:
    suffix = f'_x-dot_{dot_index_x}'

if dot_index_y is None:
    suffix += ''
else:
    suffix += f'_y-dot_{dot_index_y}'


# x_param_names = ['mu'] * 4
# y_param_names = ['t', 'l', 'b', 'd']
# common_x_axis = True
x_param_names = ['mu', 't', 'l']
y_param_names = x_param_names.copy()
common_x_axis = False

plt.rcParams['font.size'] = 20
fig, axs = plt.subplots(len(y_param_names), 2, figsize=(12, 5*len(y_param_names) + 2), sharey='row', sharex=common_x_axis)

for i, (x_param_name, y_param_name) in enumerate(zip(x_param_names, y_param_names)):

    defaults = DefaultParameters(lambda_max=0.995)

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


    # Set the root dir
    root_dir = os.path.join(save_dir, f'DiT', model_name)
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)


    loss_path = os.path.join(root_dir, loss_file)
    convergence_path = os.path.join(root_dir, convergence_file)

    tests_sub_path = os.path.join(root_dir, tests_sub_dir)     
    if not os.path.isdir(tests_sub_path):
        os.makedirs(tests_sub_path)

    filename = f'majoranization_denoising_map_{x_param_name}_{y_param_name}{suffix}'
    save_log = os.path.join(tests_sub_path, f'{filename}_log.txt')
    save_plot_path = os.path.join(tests_sub_path, f'{filename}.png')

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
        x_dataset_range=x_dataset_range,
        y_dataset_range=y_dataset_range,
        normalize_majoranization=False,
        legend=False,
        pad_inches=margin,
        fig=fig,
        axs=axs[i],
        plot_xlabel=(
            (not common_x_axis)
            or (i == len(y_param_names) - 1)
        ),
    )

plt.subplots_adjust(hspace=0.21, wspace=0.02)
# plt.subplots_adjust(hspace=0.02, wspace=0.02)

colorbar = fig.colorbar(axs[0, 1].collections[0], ax=axs, orientation='horizontal', fraction=0.07, pad=.15)
colorbar.set_label(r'$\mathcal{M}$')
colorbar.ax.tick_params(labelsize=16)
colorbar.ax.set_position([0.35, 0.17, 0.55, 0.07]) 

# Plot legend in lower left corner, but make it extra tight to not add too much space and assert that label values are not duplicated, hence plot single legend per figure and not per axis
handles, labels = axs[0, 0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
fig.legend(by_label.values(), by_label.keys(), loc='lower left', fontsize=20, frameon=False, bbox_to_anchor=(0.02, 0.185), bbox_transform=fig.transFigure)


# Add a) label over the left column and b) over the right column
axs[0,0].text(0.5, 1.1, 'a)', transform=axs[0,0].transAxes, fontsize=25, fontweight='bold', va='center', ha='center')
axs[0,1].text(0.5, 1.1, 'b)', transform=axs[0,1].transAxes, fontsize=25, fontweight='bold', va='center', ha='center')

plt.savefig(os.path.join(tests_sub_path, f'majoranization_denoising_multi_map{suffix}.png'), bbox_inches='tight', pad_inches=margin, dpi=300)
plt.close()
