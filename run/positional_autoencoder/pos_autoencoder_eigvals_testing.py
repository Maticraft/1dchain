import os

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from src.data_utils import HamiltionianDataset
from src.hamiltonian.helical_ladder import  DEFAULT_PARAMS, SpinLadder
from src.models.autoencoder import test_autoencoder
from src.models.eigvals_autoencoder import EigvalsPositionalDecoder
from src.models.autoencoder import train_autoencoder
from src.models.files import save_autoencoder_params, save_autoencoder, save_data_list, load_autoencoder_params, load_positional_autoencoder, load_ae_model
from src.plots import plot_convergence, plot_test_matrices, plot_test_eigvals
from src.models.positional_autoencoder import PositionalDecoder, PositionalEncoder

# Pretrained model
pretrained_model_dir = './autoencoder/spin_ladder/70_2_RedDist1000q_pi2delta_q/100/pretrained_gt_eigvals_positional_autoencoder_fft_tf_v4'
epoch = 4

# Paths
data_path = './data/spin_ladder/70_2_RedDist1000q_pi2delta_q'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'

# Reference eigvals plot params
eigvals_sub_dir = 'eigvals_lat_shift'
eigvals_plot_name = 'eigvals_spectre_autoencoder{}.png'
x_axis = 'q'
x_values = np.arange(0., np.pi, 0.1)
xnorm = np.pi
ylim = (-0.5, 0.5)

# Test directory name
test_dir_name = 'eigvals_tests_ep{}'

# Reference hamiltonian params
hamiltonian_sub_dir = 'hamiltonian_lat_shift'
hamiltonian_plot_name = 'hamiltonian_autoencoder{}.png'
hamiltonain_diff_plot_name = 'hamiltonian_diff{}.png'

# Load model
encoder, decoder = load_ae_model(pretrained_model_dir, epoch, PositionalEncoder, EigvalsPositionalDecoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the root dir
root_dir = os.path.join(pretrained_model_dir, test_dir_name.format(epoch))
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

eigvals_sub_path = os.path.join(root_dir, eigvals_sub_dir)     
if not os.path.isdir(eigvals_sub_path):
    os.makedirs(eigvals_sub_path)

ham_sub_path = os.path.join(root_dir, hamiltonian_sub_dir)     
if not os.path.isdir(ham_sub_path):
    os.makedirs(ham_sub_path)

test_hamiltonian = SpinLadder(**DEFAULT_PARAMS)
eigvals_path = os.path.join(eigvals_sub_path, eigvals_plot_name.format(f'_ep{epoch}'))
plot_test_eigvals(test_hamiltonian, encoder, decoder, x_axis, x_values, eigvals_path, device=device, xnorm=xnorm, ylim=ylim, decoder_eigvals=True, shift_latent=True, latent_shift=0.1)
ham_auto_path = os.path.join(ham_sub_path, hamiltonian_plot_name.format(f'_ep{epoch}' + '{}'))
ham_diff_path = os.path.join(ham_sub_path, hamiltonain_diff_plot_name.format(f'_ep{epoch}'))
plot_test_matrices(test_hamiltonian.get_hamiltonian(), encoder, decoder, save_path_rec=ham_auto_path, save_path_diff=ham_diff_path, device=device, decoder_eigvals=True, shift_latent=True, latent_shift=0.1)
