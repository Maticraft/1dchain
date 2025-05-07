import os
import pickle
from tqdm import tqdm
import numpy as np

from src.data.utils import Denormalize
from src.hamiltonian.units import AtomicUnits
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.hamiltonian.utils import plot_majorana_polarization, plot_eigvals_levels, plot_site_constant_matrix_elements, plot_interaction_constant_matrix_elements
from src.hamiltonian.quantum_dots_chain import MZM_THRESHOLD
from src.models.gan import Generator
from src.models.distribution_preserving_autoencoder import DistributionPreservingHamiltonianGenerator
from src.models.hamiltonian_generator import HamiltonianGenerator, HamiltonianGeneratorV2
from src.models.files import load_generator, load_latent_distribution, load_autoencoder_params, get_full_model_config, load_gan_submodel_state_dict, load_covariance_matrix
from src.plots import plot_generator_eigvals, plot_matrix, plot_generator_sample_eigvals_varying_property, plot_generator_sample_eigvals_increasing_property
from src.models.positional_autoencoder import PositionalDecoder, PositionalEncoder
from src.torch_utils import TorchHamiltonian


# Model params
model_name = 'QDH-1lvl-FM-pol-fixed-loss-GEN-reduced-constant-magnetic-field-delta-and-t-nonzero-potential-majoranas_distribution_preserving_autoencoder-weighted'
gen_dir = f'./gan/quantum_dots/7dots2levels_large_balanced/100/{model_name}'
data_path = './data/quantum_dots/7dots2levels_large_balanced'
data_mean_std_path = f'{data_path}/mean_std.pkl'
test_dir_name = 'generation_tests_ep{}'
# latent_distrib_dir = 'tests_majoranas_ep{}'
polarization_sub_dir = 'polarization'
hamiltonian_elements_sub_dir = 'hamiltonian_{}'
gen_epoch = 9

eigvals_transition_sub_dir = 'eigvals_transition_{}'
eigvals_gen_plot_name = 'eigvals_spectre_generator.png'
eigvals_property_varying_sub_dir = 'eigvals_property_varying_{}'
eigvals_property_varying_plot_name = 'eigvals_property_varying_{}.png'

# Eigvals plot params
num_states = 3
num_plots = 10
x_axis = 'mu'
x_values = np.linspace(-1.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-10., 10.)
ylim2 = (-5., 5.)
hopping_interaction_level = 1
vscale = 1/AtomicUnits.Eh
majorana_threshold = 20*MZM_THRESHOLD


# ------------------------------------------------------------------------------------------------------------------------------------------------
# Execute
# ------------------------------------------------------------------------------------------------------------------------------------------------
test_sub_path = os.path.join(gen_dir, test_dir_name.format(gen_epoch))    
if not os.path.isdir(test_sub_path):
    os.makedirs(test_sub_path)

mean, std = load_latent_distribution(gen_dir)
cov_matrix = load_covariance_matrix(gen_dir)

with open(data_mean_std_path, 'rb') as f:
    normalization_mean, normalization_std = pickle.load(f)

# params, encoder_params, decoder_params = load_autoencoder_params(gen_dir, PositionalEncoder, PositionalDecoder)
# generator_config = get_full_model_config(params, decoder_params)
# generator = Generator(PositionalDecoder, generator_config)
# load_gan_submodel_state_dict(gen_dir, gen_epoch, generator)

generator: Generator = load_generator(gen_dir, gen_epoch, DistributionPreservingHamiltonianGenerator)
generator.eval()

for i in tqdm(range(num_plots), desc='Plotting generator eigvals'):
    eigvals_transition_sub_path = os.path.join(test_sub_path, eigvals_transition_sub_dir.format(i))
    os.makedirs(eigvals_transition_sub_path, exist_ok=True)
    eigvals_gen_plot_path = os.path.join(eigvals_transition_sub_path, eigvals_gen_plot_name)
    properties_to_plot = {
        'potential': {'interaction_level': 0, 'part': 'all'},
        'delta': {'interaction_level': 0, 'part': 'all'},
        'magnetic_field': {'interaction_level': 0, 'part': 'all'},
        'hopping_same': {'interaction_level': 2, 'part': 'all'},
        'hopping_spin_flip': {'interaction_level': 2, 'part': 'all'},
    }
    plot_generator_eigvals(generator, num_states, eigvals_gen_plot_path, noise_type='custom', ylim=ylim, mean=mean, std=std, normalization_mean=normalization_mean, normalization_std=normalization_std, xnorm=xnorm, ynorm=ynorm, plot_properties_change=True, properties=properties_to_plot)
    
    # input_noise = generator.get_noise(1, device='cpu', noise_type='custom', mean=mean, std=std)
    input_noise = generator.get_noise(1, device='cpu', noise_type='custom', mean=mean, std=std)
    H = generator(input_noise)

    denormalization = Denormalize(normalization_mean, normalization_std)
    H = denormalization(H).squeeze()

    eigvals_property_varying_sub_path = os.path.join(test_sub_path, eigvals_property_varying_sub_dir.format(i))
    os.makedirs(eigvals_property_varying_sub_path, exist_ok=True)
    eigvals_property_varying_plot_path = os.path.join(eigvals_property_varying_sub_path, eigvals_property_varying_plot_name)
    plot_generator_sample_eigvals_varying_property(eigvals_property_varying_plot_path.format('magnetic_field'), H, property_name='magnetic_field', ylim=ylim, ynorm=ynorm)
    plot_generator_sample_eigvals_varying_property(eigvals_property_varying_plot_path.format('hopping_spin_flip'), H, property_name='hopping_spin_flip', ylim=ylim, ynorm=ynorm, offset=1)
    plot_generator_sample_eigvals_increasing_property(eigvals_property_varying_plot_path.format('substituting_potential'), H, property_name='potential', property_value_range=(-5.e-4, 5.e-4), part='real', should_replace_original_property=False,  ylim=ylim, ynorm=ynorm, offset=0)
    plot_generator_sample_eigvals_increasing_property(eigvals_property_varying_plot_path.format('substituting_delta'), H, property_name='delta', property_value_range=(-5.e-4, 5.e-4), part='both', should_replace_original_property=False,  ylim=ylim, ynorm=ynorm, offset=0)

    H = TorchHamiltonian.from_2channel_tensor(H)

    site_elements_sub_dir = os.path.join(test_sub_path, hamiltonian_elements_sub_dir.format(i))
    os.makedirs(site_elements_sub_dir, exist_ok=True)
    plot_site_constant_matrix_elements(H, 'delta', site_elements_sub_dir, ynorm=ynorm, ylim=ylim2)
    plot_site_constant_matrix_elements(H, 'potential', site_elements_sub_dir, ynorm=ynorm, ylim=ylim2)
    plot_site_constant_matrix_elements(H, 'magnetic_field', site_elements_sub_dir, ynorm=ynorm, ylim=ylim2)
    plot_interaction_constant_matrix_elements(H, 'hopping_same', site_elements_sub_dir, ynorm=ynorm, ylim=ylim2, interaction_level=hopping_interaction_level)
    plot_interaction_constant_matrix_elements(H, 'hopping_spin_flip', site_elements_sub_dir, ynorm=ynorm, ylim=ylim2, interaction_level=hopping_interaction_level)

    plot_matrix(H.get_hamiltonian().real, os.path.join(site_elements_sub_dir, f"random_hamiltonian_real.png"), vmin=-vscale, vmax=vscale)
    plot_matrix(H.get_hamiltonian().imag, os.path.join(site_elements_sub_dir, f"random_hamiltonian_imag.png"), vmin=-vscale, vmax=vscale)

    polarization_sub_path = os.path.join(site_elements_sub_dir, polarization_sub_dir)
    plot_majorana_polarization(H, polarization_sub_path, threshold = majorana_threshold, string_num=1, polaxis='x', representation=RepresentationMapping.majorana_plus_minus_up_down)
    plot_eigvals_levels(H, os.path.join(site_elements_sub_dir, 'eigvals_levels.png'), ylim=ylim, ynorm=ynorm)