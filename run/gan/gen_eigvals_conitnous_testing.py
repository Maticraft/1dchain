import os
from tqdm import tqdm

from src.majorana_utils import plot_majorana_polarization, plot_eigvals_levels
from src.models.gan import Generator
from src.models.files import load_generator, load_latent_distribution, load_autoencoder_params, get_full_model_config, load_gan_submodel_state_dict, load_covariance_matrix
from src.plots import plot_generator_eigvals
from src.models.positional_autoencoder import PositionalDecoder, PositionalEncoder
from src.torch_utils import TorchHamiltonian


# Model params
gen_dir = './gan/spin_ladder/70_2_RedDistSimplePeriodicPG/100/GAN_fft_tf_class_mvn_noise_edge_potential_0_dynamic_switch'
test_dir_name = 'generation_tests_ep{}'
latent_distrib_dir = 'tests_majoranas_ep{}'
polarization_sub_dir = 'polarization_{}'
gen_epoch = 12

eigvals_gen_plot_name = 'eigvals_spectre_generator_{}.png'


# Eigvals plot params
num_states = 3
num_plots = 10
ylim = (-0.3, 0.3)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------------------------------------------------------------------------------------
test_sub_path = os.path.join(gen_dir, test_dir_name.format(gen_epoch))    
if not os.path.isdir(test_sub_path):
    os.makedirs(test_sub_path)

mean, std = load_latent_distribution(gen_dir)
cov_matrix = load_covariance_matrix(gen_dir)

# params, encoder_params, decoder_params = load_autoencoder_params(gen_dir, PositionalEncoder, PositionalDecoder)
# generator_config = get_full_model_config(params, decoder_params)
# generator = Generator(PositionalDecoder, generator_config)
# load_gan_submodel_state_dict(gen_dir, gen_epoch, generator)

generator: Generator = load_generator(gen_dir, gen_epoch, PositionalDecoder)

for i in tqdm(range(num_plots), desc='Plotting generator eigvals'):
    eigvals_gen_plot_path = os.path.join(test_sub_path, eigvals_gen_plot_name.format(i))
    plot_generator_eigvals(generator, num_states, eigvals_gen_plot_path, noise_type='covariance', ylim=ylim, mean=mean, covariance=cov_matrix)

    # input_noise = generator.get_noise(1, device='cpu', noise_type='custom', mean=mean, std=std)
    input_noise = generator.get_noise(1, device='cpu', noise_type='covariance', mean=mean, covariance=cov_matrix)
    H = generator(input_noise).squeeze()
    H = TorchHamiltonian.from_2channel_tensor(H)

    polarization_sub_path = os.path.join(test_sub_path, polarization_sub_dir.format(i))
    plot_majorana_polarization(H, polarization_sub_path, threshold = 1.e-2, string_num=2)
    plot_eigvals_levels(H, os.path.join(test_sub_path, 'eigvals_levels_{}.png'.format(i)), ylim=ylim)