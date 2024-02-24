import os

import numpy as np

from src.hamiltonian.helical_ladder import SpinLadder, DEFAULT_PARAMS
from src.models.files import load_gan_from_positional_autoencoder, load_positional_autoencoder, load_covariance_matrix, load_latent_distribution
from src.plots import plot_test_matrices, plot_matrix


# Paths
gan_dir = './gan/spin_ladder/70_2_RedDistSimplePeriodicPG/100/GAN_fft_tf_class_mvn_noise_edge_potential_0_dynamic_switch'
# autoencoder_dir = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPG/100/twice_pretrained_positional_autoencoder_fft_tf'
test_dir_name = 'tests_ep{}'

eigvals_gan_plot_name = 'eigvals_spectre_generator_{}.png'
eigvals_diff_plot_name = 'eigvals_spectre_diff_{}.png'
eigvals_ref_plot_name = 'eigvals_spectre_ref_{}.png'

hamiltonian_gan_plot_name = 'hamiltonian_gan{}.png'
hamiltonain_diff_plot_name = 'hamiltonian_diff.png'
hamiltonian_ref_plot_name = 'hamiltonian{}.png'

random_hamiltonian_gan_plot_name = 'rand_hamiltonian{}.png'

gan_epoch = 12
# auto_epoch = 22


# Data params
# params = {'N': 70, 'M': 2, 'delta': 1.8, 'mu': 1.8, 'q': np.pi/2, 'J': 1.8, 'delta_q': np.pi, 't': 1}
params = DEFAULT_PARAMS.copy()
params['increase_potential_at_edges'] = True
params['potential_before'] = 10
params['potential_after'] = 60
params['potential'] = 5
params['periodic'] = True
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
test_sub_path = os.path.join(gan_dir, test_dir_name.format(gan_epoch))    
if not os.path.isdir(test_sub_path):
    os.makedirs(test_sub_path)

eigvals_gan_path = os.path.join(test_sub_path, eigvals_gan_plot_name.format(x_axis))
eigvals_diff_path = os.path.join(test_sub_path, eigvals_diff_plot_name.format(x_axis))
eigvals_ref_path = os.path.join(test_sub_path, eigvals_ref_plot_name.format(x_axis))
hamiltonian_gan_path = os.path.join(test_sub_path, hamiltonian_gan_plot_name)
hamiltonian_diff_path = os.path.join(test_sub_path, hamiltonain_diff_plot_name)
hamiltonian_ref_path = os.path.join(test_sub_path, hamiltonian_ref_plot_name)
random_hamiltonian_gan_path = os.path.join(test_sub_path, random_hamiltonian_gan_plot_name)

mean, std = load_latent_distribution(gan_dir)
cov_matrix = load_covariance_matrix(gan_dir)

# encoder, decoder = load_positional_autoencoder(autoencoder_dir, auto_epoch)
generator, discriminator = load_gan_from_positional_autoencoder(gan_dir, gan_epoch)

# plot_test_eigvals(SpinLadder, encoder, generator.nn, x_axis, x_values, params, eigvals_gan_path, eigvals_ref_path, eigvals_diff_path, xnorm=xnorm, ylim=ylim)
# plot_test_matrices(SpinLadder(**params).get_hamiltonian(), encoder, generator.nn, hamiltonian_diff_path, hamiltonian_gan_path, hamiltonian_ref_path)

# Generate and plot random matrix from GAN
for i in range(10):
    generator.eval()
    z = generator.get_noise(1, device='cpu', noise_type='covariance', mean=mean, covariance=cov_matrix)
    # matrix = generator(z).detach().numpy()[0]
    matrix = generator.nn(z).detach().numpy()[0]
    plot_matrix(matrix[0], random_hamiltonian_gan_path.format(f'{i}_real'))
    plot_matrix(matrix[1], random_hamiltonian_gan_path.format(f'{i}_imag'))
