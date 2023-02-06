import os

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch

from data_utils import HamiltionianDataset
from helical_ladder import  DEFAULT_PARAMS, SpinLadder
from majorana_utils import plot_autoencoder_eigvals
from models import Encoder, Decoder, train_autoencoder, test_autoencoder
from utils import save_autoencoder_params, save_autoencoder, save_data_list, plot_convergence


# Paths
dictionary_path = './data/spin_ladder/70_2/dictionary.txt'
matrices_path = './data/spin_ladder/70_2/matrices'
save_dir = './autoencoder/spin_ladder/70_2'
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
model_name = 'asymmetric_autoencoder_nearest_leaky_sf'

# Params
params = {
    'epochs': 60,
    'batch_size': 32,
    'N': 140,
    'in_channels': 2,
    'block_size': 4,
    'representation_dim': 100,
    'lr': 1.e-4,
}


# Architecture
encoder_params = {
    'kernel_size': 4,
    'kernel_size1': 4,
    'stride': 2,
    'stride1': 4,
    'dilation': 1,
    'fc_num': 4,
    'conv_num': 5,
    'kernel_num': 64,
    'kernel_num1': 32,
    'hidden_size': 128,
    'activation': 'leaky_relu',
}

decoder_params = {
    'kernel_size': 3,
    'kernel_size1': 3,
    'stride': 1,
    'dilation': 1,
    'fc_num': 4,
    'conv_num': 5,
    'kernel_num': 64,
    'hidden_size': 128,
    'upsample_method': 'nearest',
    'scale_factor': [4, 2, 2, 5/3, 21/16], # does matter only for upsample_method 'nearest' or 'bilinear'
    'activation': 'leaky_relu',
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

save_autoencoder_params(params, encoder_params, decoder_params, root_dir)

data = HamiltionianDataset(dictionary_path, matrices_path, label_idx=(1, 2))

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])

encoder = Encoder((params['in_channels'], params['N'], params['block_size']), params['representation_dim'], **encoder_params)
decoder = Decoder(params['representation_dim'], (params['in_channels'], params['N'], params['block_size']), **decoder_params)

print(encoder)
print(decoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=params['lr'])
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=params['lr'])

save_data_list(['Epoch', 'Train loss', 'Test loss'], loss_path, mode='w')

for epoch in range(1, params['epochs'] + 1):
    tr_loss = train_autoencoder(encoder, decoder, train_loader, epoch, device, encoder_optimizer, decoder_optimizer)
    te_loss = test_autoencoder(encoder, decoder, test_loader, device)
    save_autoencoder(encoder, decoder, root_dir, epoch)
    save_data_list([epoch, tr_loss, te_loss], loss_path)

    eigvals_path = os.path.join(eigvals_sub_path, eigvals_plot_name.format(f'_ep{epoch}'))
    plot_autoencoder_eigvals(SpinLadder, encoder, decoder, x_axis, x_values, DEFAULT_PARAMS, eigvals_path, device=device, xnorm=xnorm, ylim=ylim)
plot_convergence(loss_path, convergence_path, read_label=True)
