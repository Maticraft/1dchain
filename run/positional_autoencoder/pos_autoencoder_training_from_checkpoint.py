import os

import sys
sys.path.append('.')

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from src.data_utils import HamiltionianDataset
from src.hamiltonian.helical_ladder import  DEFAULT_PARAMS, SpinLadder
from src.models.autoencoder import test_autoencoder
from src.models.hamiltonian_generator import HamiltonianGeneratorV2, HamiltonianGenerator
from src.models.autoencoder import train_autoencoder
from src.models.eigvals_autoencoder import EigvalsPositionalDecoder
from src.models.files import save_autoencoder_params, save_autoencoder, save_data_list, load_autoencoder_params, load_positional_autoencoder, load_ae_model
from src.plots import plot_convergence, plot_test_matrices, plot_test_eigvals
from src.models.positional_autoencoder import PositionalDecoder, PositionalEncoder

# Pretrained model
pretrained_model_dir = './autoencoder/spin_ladder/70_2_RedDist1000q_pi2delta_q/100/pretrained_pos_encoder_symmetric_hamiltonian_generator_v2_varying_potential_and_delta_tf'
epoch = 10

# Paths
data_path = './data/spin_ladder/70_2_RedDistSimplePeriodicPGSeparatedMajoranas'
save_dir = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPGSeparatedMajoranas'
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
model_name = 'twice_pretrained_pos_encoder_symmetric_hamiltonian_generator_v2_varying_potential_and_delta_tf'

# Load model
encoder, decoder = load_ae_model(pretrained_model_dir, epoch, PositionalEncoder, HamiltonianGeneratorV2)
params, encoder_params, decoder_params = load_autoencoder_params(pretrained_model_dir, PositionalEncoder, HamiltonianGeneratorV2)

# Modify params
params['learning_rate'] = 1.e-5
params['epochs'] = 40
params['det_loss'] = False
params['det_loss_weight'] = 0.

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

data = HamiltionianDataset(data_path, label_idx=(3, 4), eigvals=(params['eigenvalues_loss'] or params['gt_eigvals']), eig_decomposition=params['eigenstates_loss'], format='csr', eigvals_num=params['eigvals_num'])

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=params['lr'])
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=params['lr'])
encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min')
decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min')

save_data_list(['Epoch', 'Train loss', 'Train edge loss', 'Train eigenvalues loss', 'Train eigenstates loss', 'Train diag loss', 'Train determinant loss', 'Test loss', 'Test edge loss', 'Test eigevalues loss', 'Test eigenstates loss', 'Test diag loss', 'Test determinant loss'], loss_path, mode='w')

for epoch in range(1, params['epochs'] + 1):
    tr_loss, tr_edge_loss, tr_ev_loss, tr_eig_loss, tr_diag_loss, tr_det_loss = train_autoencoder(
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
        det_loss=params['det_loss'],
        det_loss_weight=params['det_loss_weight'],
        gt_eigvals=params['gt_eigvals']
    )
    te_loss, te_edge_loss, te_ev_loss, te_eig_loss, te_diag_loss, te_det_loss = test_autoencoder(
        encoder,
        decoder,
        test_loader,
        device,
        edge_loss=params['edge_loss'],
        eigenvalues_loss=params['eigenvalues_loss'],
        eigenstates_loss=params['eigenstates_loss'],
        diag_loss=params['diag_loss'],
        det_loss=params['det_loss'],
        gt_eigvals=params['gt_eigvals']
    )
    encoder_scheduler.step(te_loss)
    decoder_scheduler.step(te_loss)

    save_autoencoder(encoder, decoder, root_dir, epoch)
    save_data_list([epoch, tr_loss, tr_edge_loss, tr_ev_loss, tr_eig_loss, tr_diag_loss, tr_det_loss, te_loss, te_edge_loss, te_ev_loss, te_eig_loss, te_diag_loss, te_det_loss], loss_path)

    eigvals_path = os.path.join(eigvals_sub_path, eigvals_plot_name.format(f'_ep{epoch}'))
    plot_test_eigvals(SpinLadder, encoder, decoder, x_axis, x_values, DEFAULT_PARAMS, eigvals_path, device=device, xnorm=xnorm, ylim=ylim, decoder_eigvals=params['gt_eigvals'])
    ham_auto_path = os.path.join(ham_sub_path, hamiltonian_plot_name.format(f'_ep{epoch}' + '{}'))
    ham_diff_path = os.path.join(ham_sub_path, hamiltonain_diff_plot_name.format(f'_ep{epoch}'))
    plot_test_matrices(SpinLadder(**DEFAULT_PARAMS).get_hamiltonian(), encoder, decoder, save_path_rec=ham_auto_path, save_path_diff=ham_diff_path, device=device, decoder_eigvals=params['gt_eigvals'])

plot_convergence(loss_path, convergence_path, read_label=True)
