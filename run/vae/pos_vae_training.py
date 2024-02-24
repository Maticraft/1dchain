import os

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from src.data_utils import HamiltionianDataset
from src.hamiltonian.helical_ladder import  DEFAULT_PARAMS, SpinLadder
from src.models.autoencoder import test_autoencoder
from src.models.vae import VariationalPositionalEncoder
from src.models.vae import train_vae
from src.models.files import save_autoencoder_params, save_autoencoder, save_data_list
from src.plots import plot_convergence, plot_test_matrices, plot_test_eigvals
from src.models.positional_autoencoder import PositionalDecoder

# Paths
data_path = './data/spin_ladder/70_2_RedDistFixed'
save_dir = './vae/spin_ladder/70_2_RedDistFixed'
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


# Model name
model_name = 'positional_vae_fft_tf_no_kl'

# Params
params = {
    'epochs': 60,
    'batch_size': 64,
    'N': 140,
    'in_channels': 10,
    'block_size': 4,
    'representation_dim': 100,
    'lr': 1.e-5,
    'edge_loss': False,
    'edge_loss_weight': 1.,
    'eigenstates_loss': False,
    'eigenstates_loss_weight': 1.,
    'diag_loss': True,
    'diag_loss_weight': 0.01,
}

# Architecture
encoder_params = {
    'kernel_num': 64,
    'kernel_size': 4,
    'activation': 'leaky_relu',
    'freq_enc_depth': 4,
    'freq_enc_hidden_size': 128,
    'block_enc_depth': 4,
    'block_enc_hidden_size': 128,
    'padding_mode': 'zeros',
}


decoder_params = {
    "kernel_num": 64,
    "activation": "leaky_relu",
    "freq_dec_depth": 4,
    "freq_dec_hidden_size": 128,
    "block_dec_depth": 4,
    "block_dec_hidden_size": 128,
    "seq_dec_depth": 4,
    "seq_dec_hidden_size": 128,
}


# Set the root dir
root_dir = os.path.join(save_dir, f'{params["representation_dim"]}', model_name)
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

loss_path = os.path.join(root_dir, loss_file)
convergence_path = os.path.join(root_dir, convergence_file)

eigvals_sub_path = os.path.join(root_dir, eigvals_sub_dir)     
if not os.path.isdir(eigvals_sub_path):
    os.makedirs(eigvals_sub_path)

ham_sub_path = os.path.join(root_dir, hamiltonian_sub_dir)     
if not os.path.isdir(ham_sub_path):
    os.makedirs(ham_sub_path)

save_autoencoder_params(params, encoder_params, decoder_params, root_dir)

data = HamiltionianDataset(data_path, label_idx=(3, 4), eig_decomposition=params['eigenstates_loss'], format='numpy')

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])

encoder = VariationalPositionalEncoder((params['in_channels'], params['N'], params['block_size']), params['representation_dim'], **encoder_params)
decoder = PositionalDecoder(params['representation_dim'], (params['in_channels'], params['N'], params['block_size']), **decoder_params)

print(encoder)
print(decoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=params['lr'])
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=params['lr'])

save_data_list(['Epoch', 'Train rec loss', 'Train kl loss', 'Train edge loss', 'Train eigenstates loss', 'Train diag loss', 'Test loss', 'Test edge loss', 'Test eigenstates loss', 'Test diag loss'], loss_path, mode='w')

for epoch in range(1, params['epochs'] + 1):
    tr_loss, tr_kl_loss, tr_edge_loss, tr_eig_loss, tr_diag_loss = train_vae(
        encoder,
        decoder,
        train_loader,
        epoch,
        device,
        encoder_optimizer,
        decoder_optimizer,
        edge_loss=params['edge_loss'],
        edge_loss_weight=params['edge_loss_weight'],
        eigenstates_loss=params['eigenstates_loss'],
        eigenstates_loss_weight=params['eigenstates_loss_weight'],
        diag_loss=params['diag_loss'],
        diag_loss_weight=params['diag_loss_weight'],
    )
    te_loss, te_edge_loss, te_eig_loss, te_diag_loss = test_autoencoder(encoder, decoder, test_loader, device, edge_loss=params['edge_loss'], eigenstates_loss=params['eigenstates_loss'], diag_loss=params['diag_loss'])
    save_autoencoder(encoder, decoder, root_dir, epoch)
    save_data_list([epoch, tr_loss, tr_kl_loss, tr_edge_loss, tr_eig_loss, tr_diag_loss, te_loss, te_edge_loss, te_eig_loss, te_diag_loss], loss_path)

    test_hamiltonian = SpinLadder(**DEFAULT_PARAMS)
    eigvals_path = os.path.join(eigvals_sub_path, eigvals_plot_name.format(f'_ep{epoch}'))
    plot_test_eigvals(test_hamiltonian, encoder, decoder, x_axis, x_values, eigvals_path, device=device, xnorm=xnorm, ylim=ylim)
    ham_auto_path = os.path.join(ham_sub_path, hamiltonian_plot_name.format(f'_ep{epoch}' + '{}'))
    ham_diff_path = os.path.join(ham_sub_path, hamiltonain_diff_plot_name.format(f'_ep{epoch}'))
    plot_test_matrices(test_hamiltonian.get_hamiltonian(), encoder, decoder, save_path_rec=ham_auto_path, save_path_diff=ham_diff_path, device=device)
   
plot_convergence(loss_path, convergence_path, read_label=True)
