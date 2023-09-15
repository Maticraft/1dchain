import os

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from data_utils import HamiltionianDataset
from helical_ladder import  DEFAULT_PARAMS, SpinLadder
from models import Generator, PositionalEncoder, PositionalDecoder, Classifier
from models_utils import train_noise_controller, test_generator_with_classifier, test_noise_controller
from models_files import get_full_model_config, load_gan_submodel_state_dict, load_latent_distribution, save_gan_params, save_latent_distribution, save_model, save_data_list, save_model, load_autoencoder_params, load_classifier, load_positional_autoencoder, save_autoencoder_params
from models_plots import plot_convergence, plot_matrix

# Pretrained model
pretrained_model_dir = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPG/100/classifier_bal_twice_pretrained_positional_autoencoder_fft_tf'
epoch = 4
distibution_path = os.path.join(pretrained_model_dir, 'tests_ep{}'.format(epoch))


# Paths
data_path = './data/spin_ladder/70_2_RedDistSimplePeriodicPG'
save_dir = './gan/spin_ladder/70_2_RedDistSimplePeriodicPG'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'

# Reference eigvals plot params
eigvals_sub_dir = 'eigvals'
eigvals_plot_name = 'eigvals_spectre_autoencoder{}.png'
x_axis = 'q'
x_values = np.arange(0., np.pi, 0.1)
xnorm = np.pi
ylim = (-0.5, 0.5)

# Reference hamiltonian params
hamiltonian_sub_dir = 'hamiltonian'
hamiltonian_plot_name = 'hamiltonian_autoencoder{}.png'
hamiltonain_diff_plot_name = 'hamiltonian_diff{}.png'

# New model name
model_name = 'gen_ae_fft_tf_polarization_fixed2_full_model'

# Load params
params, encoder_params, decoder_params = load_autoencoder_params(pretrained_model_dir, PositionalEncoder, PositionalDecoder)

# Modify params
params['learning_rate'] = 1.e-5
params['diag_loss'] = True
params['diag_loss_weight'] = 0.01
params['log_scaled_loss'] = False
params['eigenstates_loss'] = False
params['eigenstates_loss_weight'] = 1.
params['epochs'] = 20

# Set the root dir
root_dir = os.path.join(save_dir, f'{params["representation_dim"]}', model_name)
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

init_distribution = load_latent_distribution(distibution_path)
save_latent_distribution(init_distribution, root_dir)

# Load model
generator_config = get_full_model_config(params, decoder_params)
generator = Generator(PositionalDecoder, generator_config, distribution=init_distribution)
load_gan_submodel_state_dict(pretrained_model_dir, epoch, generator)

encoder, _ = load_positional_autoencoder(pretrained_model_dir, epoch)

classifier = load_classifier(pretrained_model_dir, epoch, input_dim = params['representation_dim'])

loss_path = os.path.join(root_dir, loss_file)
convergence_path = os.path.join(root_dir, convergence_file)

eigvals_sub_path = os.path.join(root_dir, eigvals_sub_dir)     
if not os.path.isdir(eigvals_sub_path):
    os.makedirs(eigvals_sub_path)

ham_sub_path = os.path.join(root_dir, hamiltonian_sub_dir)     
if not os.path.isdir(ham_sub_path):
    os.makedirs(ham_sub_path)

save_gan_params(params, decoder_params, encoder_params, root_dir)
save_autoencoder_params(params, encoder_params, decoder_params, root_dir)

data = HamiltionianDataset(data_path, label_idx=(3, 4), eig_decomposition=params['eigenstates_loss'], format='csr')

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

noise_controller_optimizer = torch.optim.Adam(generator.parameters(), lr=params['learning_rate'])

save_data_list(['Epoch', 'Train_loss', 'Train_distribution_loss', 'Test_loss', 'Test_classifier_acc'], loss_path, mode='w')

for epoch in range(1, params['epochs'] + 1):
    tr_loss, tr_distr_loss = train_noise_controller(generator, train_loader, epoch, device, noise_controller_optimizer, init_distribution)
    te_loss = test_noise_controller(generator, test_loader, epoch, device, init_distribution)

    save_model(generator, root_dir, epoch)
    save_model(classifier, root_dir, epoch)
    save_data_list([epoch, tr_loss, tr_distr_loss, te_loss], loss_path)

    # Plot sample hamiltonian
    test_matrix_path = os.path.join(root_dir, f'test_{epoch}')
    generator.to(device)
    os.makedirs(test_matrix_path, exist_ok=True)
    for i in range(10):
        generator.eval()
        z = generator.get_noise(1, device=device, noise_type='hybrid')
        matrix = generator.nn(z).detach().cpu().numpy()[0]
        
        plot_matrix(matrix[0], os.path.join(test_matrix_path, f"random_hamiltonian_real_{i}.png"))
        plot_matrix(matrix[1], os.path.join(test_matrix_path, f"random_hamiltonian_imag_{i}.png"))

plot_convergence(loss_path, convergence_path, read_label=True)
