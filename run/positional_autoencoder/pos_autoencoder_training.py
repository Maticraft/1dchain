import os

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from src.data_utils import HamiltionianDataset
from src.hamiltonian.helical_ladder import  DEFAULT_PARAMS, SpinLadder
from src.models.hamiltonian_generator import HamiltonianGenerator, HamiltonianGeneratorV2
from src.models.autoencoder import train_autoencoder
from src.models.autoencoder import Decoder, Encoder, test_autoencoder
from src.models.eigvals_autoencoder import EigvalsPositionalDecoder, EigvalsPositionalEncoder
from src.models.files import save_autoencoder_params, save_autoencoder, save_data_list
from src.plots import plot_convergence, plot_test_matrices, plot_test_eigvals
from src.models.positional_autoencoder import PositionalDecoder, PositionalEncoder

# Paths
data_path = './data/spin_ladder/70_2_RedDistFixed'
save_dir = './autoencoder/spin_ladder/70_2_RedDistFixed'
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
model_name = 'pos_encoder_hamiltonian_generator_v2_tf'

# Params
params = {
    'epochs': 60,
    'batch_size': 64,
    'N': 140,
    'in_channels': 10,
    'block_size': 4,
    'representation_dim': 100,
    'eigvals_num': 560,
    'lr': 1.e-5,
    'edge_loss': False,
    'edge_loss_weight': 1.,
    'eigenvalues_loss': False,
    'eigenvalues_loss_weight': 1.,
    'eigenstates_loss': False,
    'eigenstates_loss_weight': 1.,
    'diag_loss': True,
    'diag_loss_weight': 0.01,
    'log_scaled_loss': False,
    'gt_eigvals': False
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
    'mlp_layers': 3,
}


decoder_params = {
    # 'kernel_num': 64,
    'activation': 'leaky_relu',
    'freq_dec_depth': 4,
    'freq_dec_hidden_size': 128,
    'block_dec_depth': 4,
    'block_dec_hidden_size': 128,
    'seq_dec_depth': 4,
    'seq_dec_hidden_size': 128,
    'reduce_blocks': False,
    'seq_num': 32,
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

data = HamiltionianDataset(data_path, label_idx=(3, 4), eigvals=(params['eigenvalues_loss'] or params['gt_eigvals']), eig_decomposition=params['eigenstates_loss'], format='numpy', eigvals_num=params['eigvals_num'])

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])

encoder = PositionalEncoder((params['in_channels'], params['N'], params['block_size']), params['representation_dim'], **encoder_params)
# decoder = EigvalsPositionalDecoder(params['representation_dim'], (params['in_channels'], params['N'], params['block_size']), **decoder_params)
decoder = HamiltonianGeneratorV2(params['representation_dim'], (params['in_channels'], params['N'], params['block_size']), **decoder_params)

print(encoder)
print(decoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=params['lr'])
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=params['lr'])

save_data_list(['Epoch', 'Train loss', 'Train edge loss', 'Train eigenvalues loss', 'Train eigenstates loss', 'Train diag loss', 'Test loss', 'Test edge loss', 'Test eigevalues loss', 'Test eigenstates loss', 'Test diag loss'], loss_path, mode='w')

for epoch in range(1, params['epochs'] + 1):
    tr_loss, tr_edge_loss, tr_ev_loss, tr_eig_loss, tr_diag_loss, _ = train_autoencoder(
        encoder,
        decoder,
        train_loader,
        epoch,
        device,
        encoder_optimizer,
        decoder_optimizer,
        edge_loss=params['edge_loss'],
        edge_loss_weight=params['edge_loss_weight'],
        eigenvalues_loss=params['eigenvalues_loss'],
        eigenvalues_loss_weight=params['eigenvalues_loss_weight'],
        eigenstates_loss=params['eigenstates_loss'],
        eigenstates_loss_weight=params['eigenstates_loss_weight'],
        diag_loss=params['diag_loss'],
        diag_loss_weight=params['diag_loss_weight'],
        log_scaled_loss=params['log_scaled_loss'],
        gt_eigvals=params['gt_eigvals']
    )
    te_loss, te_edge_loss, te_ev_loss, te_eig_loss, te_diag_loss, _ = test_autoencoder(
        encoder,
        decoder,
        test_loader,
        device,
        edge_loss=params['edge_loss'],
        eigenvalues_loss=params['eigenvalues_loss'],
        eigenstates_loss=params['eigenstates_loss'],
        diag_loss=params['diag_loss'],
        gt_eigvals=params['gt_eigvals']
    )
    save_autoencoder(encoder, decoder, root_dir, epoch)
    save_data_list([epoch, tr_loss, tr_edge_loss, tr_ev_loss, tr_eig_loss, tr_diag_loss, te_loss, te_edge_loss, te_ev_loss, te_eig_loss, te_diag_loss], loss_path)

    eigvals_path = os.path.join(eigvals_sub_path, eigvals_plot_name.format(f'_ep{epoch}'))
    plot_test_eigvals(SpinLadder, encoder, decoder, x_axis, x_values, DEFAULT_PARAMS, eigvals_path, device=device, xnorm=xnorm, ylim=ylim, decoder_eigvals=(params['eigenvalues_loss'] or params['gt_eigvals']))
    ham_auto_path = os.path.join(ham_sub_path, hamiltonian_plot_name.format(f'_ep{epoch}' + '{}'))
    ham_diff_path = os.path.join(ham_sub_path, hamiltonain_diff_plot_name.format(f'_ep{epoch}'))
    plot_test_matrices(SpinLadder(**DEFAULT_PARAMS).get_hamiltonian(), encoder, decoder, save_path_rec=ham_auto_path, save_path_diff=ham_diff_path, device=device, decoder_eigvals=(params['eigenvalues_loss'] or params['gt_eigvals']))
   
plot_convergence(loss_path, convergence_path, read_label=True)
