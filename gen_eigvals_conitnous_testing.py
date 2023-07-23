import os
from tqdm import tqdm

from majorana_utils import plot_majorana_polarization
from models import Generator, PositionalDecoder, PositionalEncoder
from models_files import load_generator, load_latent_distribution, load_autoencoder_params, get_full_model_config, load_gan_submodel_state_dict
from models_plots import plot_generator_eigvals
from torch_hamiltonian import TorchHamiltonian


# Model params
gen_dir = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPG/100/classifier_bal_twice_pretrained_positional_autoencoder_fft_tf'
test_dir_name = 'generation_tests_ep{}'
latent_distrib_dir = 'tests_majoranas_ep{}'
polarization_sub_dir = 'polarization_{}'
gen_epoch = 4

eigvals_gen_plot_name = 'eigvals_spectre_generator_{}.png'


# Eigvals plot params
num_states = 3
num_plots = 10
ylim = (-0.5, 0.5)


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------------------------------------------------------------------------------------
test_sub_path = os.path.join(gen_dir, test_dir_name.format(gen_epoch))    
if not os.path.isdir(test_sub_path):
    os.makedirs(test_sub_path)

latent_space_sub_path = os.path.join(gen_dir, latent_distrib_dir.format(gen_epoch))
mean, std = load_latent_distribution(latent_space_sub_path)

params, encoder_params, decoder_params = load_autoencoder_params(gen_dir, PositionalEncoder, PositionalDecoder)
generator_config = get_full_model_config(params, decoder_params)
generator = Generator(PositionalDecoder, generator_config)
load_gan_submodel_state_dict(gen_dir, gen_epoch, generator)


for i in tqdm(range(num_plots), desc='Plotting generator eigvals'):
    eigvals_gen_plot_path = os.path.join(test_sub_path, eigvals_gen_plot_name.format(i))
    plot_generator_eigvals(generator, num_states, eigvals_gen_plot_path, noise_type='custom', ylim=ylim, mean=mean, std=std)

    input_noise = generator.get_noise(1, device='cpu', noise_type='custom', mean=mean, std=std)
    H = generator(input_noise).squeeze()
    H = TorchHamiltonian.from_2channel_tensor(H)

    polarization_sub_path = os.path.join(test_sub_path, polarization_sub_dir.format(i))
    plot_majorana_polarization(H, polarization_sub_path, threshold = 1.e-2, string_num=2)