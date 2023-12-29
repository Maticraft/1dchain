import os

import numpy as np
import torch

from src.hamiltonian.helical_ladder import SpinLadder, DEFAULT_PARAMS
from src.hamiltonian.utils import plot_eigvals, plot_site_matrix_elements
from src.models.positional_autoencoder import PositionalEncoder
from src.models.hamiltonian_generator import HamiltonianGenerator
from src.models.files import load_autoencoder, load_positional_autoencoder, load_ae_model
from src.models.utils import reconstruct_hamiltonian
from src.torch_utils import TorchHamiltonian
from src.plots import plot_test_matrices, plot_test_eigvals


# Paths
autoencoder_dir = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPGBalancedZM/100/twice_pretrained_pos_encoder_hamiltonian_generator_tf'
test_dir_name = 'ae_tests_ep{}_v2'
ae_hamiltonian_params_dir = 'ae_hamiltonian_params'
ref_hamiltonian_params_dir = 'ref_hamiltonian_params'

eigvals_auto_plot_name = 'eigvals_spectre_autoencoder_{}.png'
eigvals_diff_plot_name = 'eigvals_spectre_diff_{}.png'
eigvals_ref_plot_name = 'eigvals_spectre_ref_{}.png'

hamiltonian_auto_plot_name = 'hamiltonian_autoencoder{}.png'
hamiltonain_diff_plot_name = 'hamiltonian_diff.png'
hamiltonian_ref_plot_name = 'hamiltonian{}.png'

epoch = 22


# Data params
# params = {'N': 70, 'M': 2, 'delta': 1.8, 'mu': 1.8, 'q': np.pi/2, 'J': 1.8, 'delta_q': np.pi, 't': 1}
params = DEFAULT_PARAMS.copy()
params['increase_potential_at_edges'] = False
params['potential_before'] = 10
params['potential_after'] = 60
params['potential'] = 5
params['periodic'] = False
params['use_disorder'] = False
params['disorder_potential'] = 5
params['disorder_positions'] = [{'i': 10, 'j': 0}, {'i': 30, 'j': 0}, {'i': 30, 'j': 1}, {'i': 50, 'j': 1}]


# Eigvals plot params
x_axis = 'q'
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

encoder, decoder = load_ae_model(autoencoder_dir, epoch, PositionalEncoder, HamiltonianGenerator)

# plot_test_eigvals(SpinLadder, encoder, decoder, x_axis, x_values, params, eigvals_auto_path, eigvals_ref_path, eigvals_diff_path, xnorm=xnorm, ylim=ylim)

hamiltonian = SpinLadder(**params)
hamiltonian_matrix = hamiltonian.get_hamiltonian()
# plot_test_matrices(hamiltonian_matrix, encoder, decoder, hamiltonian_diff_path, hamiltonian_auto_path, hamiltonian_ref_path)

reconstructed_hamiltonian_matrix = reconstruct_hamiltonian(hamiltonian_matrix, encoder, decoder)
torch_hamiltonian = TorchHamiltonian(torch.from_numpy(reconstructed_hamiltonian_matrix))

ae_hamiltonian_elements_dir = os.path.join(test_sub_path, ae_hamiltonian_params_dir)
os.makedirs(ae_hamiltonian_elements_dir, exist_ok=True)
plot_site_matrix_elements(torch_hamiltonian, 'potential', ae_hamiltonian_elements_dir)
plot_site_matrix_elements(torch_hamiltonian, 'delta', ae_hamiltonian_elements_dir)
plot_site_matrix_elements(torch_hamiltonian, 'spin', ae_hamiltonian_elements_dir)
plot_site_matrix_elements(torch_hamiltonian, 'interaction_i_j', ae_hamiltonian_elements_dir)
plot_site_matrix_elements(torch_hamiltonian, 'interaction_j_i', ae_hamiltonian_elements_dir)

ref_hamiltonian_elements_dir = os.path.join(test_sub_path, ref_hamiltonian_params_dir)
os.makedirs(ref_hamiltonian_elements_dir, exist_ok=True)
plot_site_matrix_elements(hamiltonian, 'potential', ref_hamiltonian_elements_dir)
plot_site_matrix_elements(hamiltonian, 'delta', ref_hamiltonian_elements_dir)
plot_site_matrix_elements(hamiltonian, 'spin', ref_hamiltonian_elements_dir)
plot_site_matrix_elements(hamiltonian, 'interaction_i_j', ref_hamiltonian_elements_dir)
plot_site_matrix_elements(hamiltonian, 'interaction_j_i', ref_hamiltonian_elements_dir)
