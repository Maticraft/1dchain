import os

import numpy as np

from helical_ladder import SpinLadder, DEFAULT_PARAMS
from majorana_utils import plot_eigvals
from models_files import load_autoencoder, load_positional_autoencoder
from models_plots import plot_test_matrices, plot_test_eigvals


# Paths
autoencoder_dir = './autoencoder/spin_ladder/70_2_RedDist1000q_pi2delta_q/100/pretrained_positional_autoencoder_fft_lstm_v2-2'
test_dir_name = 'tests_ep{}'

eigvals_auto_plot_name = 'eigvals_spectre_autoencoder_{}.png'
eigvals_diff_plot_name = 'eigvals_spectre_diff_{}.png'
eigvals_ref_plot_name = 'eigvals_spectre_ref_{}.png'

hamiltonian_auto_plot_name = 'hamiltonian_autoencoder{}.png'
hamiltonain_diff_plot_name = 'hamiltonian_diff.png'
hamiltonian_ref_plot_name = 'hamiltonian{}.png'

epoch = 8

# Data params
# params = {'N': 70, 'M': 2, 'delta': 1.8, 'mu': 1.8, 'q': np.pi/2, 'J': 1.8, 'delta_q': np.pi, 't': 1}
params = DEFAULT_PARAMS

# Eigvals plot params
x_axis = 'delta_q'
x_values = np.concatenate((np.arange(0., np.pi, 2*np.pi / 100), np.arange(np.pi, 2*np.pi, 2*np.pi / 100)))
# x_values = np.arange(0., 4., 0.1)

xnorm = np.pi
ylim = (-2, 2) 


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------------------------------------------------------------------------------------
test_sub_path = os.path.join(autoencoder_dir, test_dir_name.format(epoch))    
if not os.path.isdir(test_sub_path):
    os.makedirs(test_sub_path)

eigvals_auto_path = os.path.join(test_sub_path, eigvals_auto_plot_name.format(x_axis))
eigvals_diff_path = os.path.join(test_sub_path, eigvals_diff_plot_name.format(x_axis))
eigvals_ref_path = os.path.join(test_sub_path, eigvals_ref_plot_name.format(x_axis))
hamiltonian_auto_path = os.path.join(test_sub_path, hamiltonian_auto_plot_name)
hamiltonian_diff_path = os.path.join(test_sub_path, hamiltonain_diff_plot_name)
hamiltonian_ref_path = os.path.join(test_sub_path, hamiltonian_ref_plot_name)


encoder, decoder = load_positional_autoencoder(autoencoder_dir, epoch)
plot_test_eigvals(SpinLadder, encoder, decoder, x_axis, x_values, params, eigvals_auto_path, eigvals_ref_path, eigvals_diff_path, xnorm=xnorm, ylim=ylim)
plot_test_matrices(SpinLadder(**params).get_hamiltonian(), encoder, decoder, hamiltonian_diff_path, hamiltonian_auto_path, hamiltonian_ref_path)
