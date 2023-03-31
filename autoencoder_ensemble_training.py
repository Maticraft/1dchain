import os

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from data_utils import HamiltionianDataset
from helical_ladder import  DEFAULT_PARAMS, SpinLadder
from models import EncoderEnsemble, DecoderEnsemble
from models_utils import train_autoencoder, test_autoencoder
from models_files import save_autoencoder_params, save_autoencoder, save_data_list
from models_plots import plot_convergence, plot_test_eigvals


# Paths
dictionary_path = './data/spin_ladder/70_2/dictionary.txt'
matrices_path = './data/spin_ladder/70_2/matrices'
save_dir = './autoencoder_ensemble/spin_ladder/70_2'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'


# Reference eigvals plot params
eigvals_sub_dir = 'eigvals'
eigvals_plot_name = 'eigvals_spectre_autoencoder{}.png'
x_axis = 'q'
x_values = np.arange(0., np.pi, 0.1)
xnorm = np.pi
ylim = (-0.5, 0.5)


# Model name
model_name = 'hybrid_ensemble_symmetric_autoencoder_edge_loss1'

# Params
params = {
    'epochs': 120,
    'batch_size': 64,
    'N': 140,
    'in_channels': 2,
    'block_size': 4,
    'representation_dim': 100,
    'lr': 1.e-5,
    'edge_loss': True,
    'edge_loss_weight': 1.,
}


# Architecture
encoder0_params = {
    'kernel_size': 2,
    'kernel_size1': 35,
    'stride': 2,
    'stride1': 1,
    'dilation': 1,
    'dilation1': 16,
    'fc_num': 4,
    'conv_num': 3,
    'kernel_num': 128,
    'kernel_num1': 64,
    'hidden_size': 256,
    'activation': 'leaky_relu',
}

encoder1_params = {
    "kernel_size": 4,
    "kernel_size1": 4,
    "stride": 2,
    "stride1": 4,
    "dilation": 1,
    "fc_num": 4,
    "conv_num": 5,
    "kernel_num": 64,
    "kernel_num1": 32,
    "hidden_size": 128,
    "activation": "leaky_relu"
}

encoders_params = {
    'encoders_num': 2,
    'edge_encoder_idx': None,
    'encoder_0': encoder0_params,
    'encoder_1': encoder1_params,
}

decoder0_params = {
    'kernel_size': 2,
    'kernel_size1': 35,
    'stride': 2,
    'stride1': 1,
    'dilation': 1,
    'dilation1': 16,
    'fc_num': 4,
    'conv_num': 3,
    'kernel_num': 128,
    'kernel_num1': 64,
    'hidden_size': 256,
    'upsample_method': 'transpose',
    'scale_factor': 2, # does matter only for upsample_method 'nearest' or 'bilinear'
    'activation': 'leaky_relu',
}

decoder1_params = {
    "kernel_size": 3,
    "kernel_size1": 3,
    "stride": 1,
    "dilation": 1,
    "fc_num": 4,
    "conv_num": 5,
    "kernel_num": 64,
    "hidden_size": 128,
    "upsample_method": "nearest",
    "scale_factor": [
        4,
        2,
        2,
        1.6666666666666667,
        1.3125
    ],
    "activation": "leaky_relu"
}

decoders_params = {
    'decoders_num': 2,
    'edge_decoder_idx': None,
    'decoder_0': decoder0_params,
    'decoder_1': decoder1_params
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

save_autoencoder_params(params, encoders_params, decoders_params, root_dir)

data = HamiltionianDataset(dictionary_path, matrices_path, label_idx=(1, 2))

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])

encoder = EncoderEnsemble((params['in_channels'], params['N'], params['block_size']), params['representation_dim'], encoders_params)
decoder = DecoderEnsemble(params['representation_dim'], (params['in_channels'], params['N'], params['block_size']), decoders_params)

print(encoder)
print(decoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=params['lr'])
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=params['lr'])

save_data_list(['Epoch', 'Train loss', 'Train edge loss', 'Train eigenstates loss', 'Test loss', 'Test edge loss', 'Test eigenstates loss'], loss_path, mode='w')

for epoch in range(1, params['epochs'] + 1):
    tr_loss, tr_edge_loss, tr_eig_loss = train_autoencoder(
        encoder,
        decoder,
        train_loader,
        epoch,
        device,
        encoder_optimizer,
        decoder_optimizer,
        edge_loss=params['edge_loss'],
        edge_loss_weight=params['edge_loss_weight']
    )
    te_loss, te_edge_loss, te_eig_loss = test_autoencoder(encoder, decoder, test_loader, device, edge_loss=params['edge_loss'])
    save_autoencoder(encoder, decoder, root_dir, epoch)
    save_data_list([epoch, tr_loss, tr_edge_loss, tr_eig_loss, te_loss, te_edge_loss, te_eig_loss], loss_path)

    eigvals_path = os.path.join(eigvals_sub_path, eigvals_plot_name.format(f'_ep{epoch}'))
    plot_test_eigvals(SpinLadder, encoder, decoder, x_axis, x_values, DEFAULT_PARAMS, eigvals_path, device=device, xnorm=xnorm, ylim=ylim)
plot_convergence(loss_path, convergence_path, read_label=True)
