import os

from torch.utils.data import random_split, DataLoader
import torch

from data_utils import HamiltionianDataset
from models import Encoder, Decoder, train_autoencoder, test_autoencoder
from utils import save_autoencoder_params, save_autoencoder, save_data_list, plot_convergence


# Paths
dictionary_path = './data/spin_ladder/70_2/dictionary.txt'
matrices_path = './data/spin_ladder/70_2/matrices'
save_dir = './autoencoder/spin_ladder/70_2'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'

# Model name
model_name = 'symmetric_autoencoder'

# Params
params = {
    'epochs': 10,
    'batch_size': 512,
    'N': 140,
    'in_channels': 2,
    'block_size': 4,
    'representation_dim': 10,
    'lr': 1.e-5,
}


# Architecture
encoder_params = {
    'kernel_size': params['block_size'],
    'stride': params['block_size'],
    'dilation': 1,
    'fc_num': 1,
    'conv_num': 1,
    'kernel_num': 16,
    'hidden_size': 32,
}

decoder_params = {
    'kernel_size': params['block_size'],
    'stride': params['block_size'],
    'dilation': 1,
    'fc_num': 1,
    'conv_num': 1,
    'kernel_num': 16,
    'hidden_size': 32,
    'upsample_method': 'transpose',
    'scale_factor': 2*params['block_size'], # does matter only for upsample_method 'nearest' or 'bilinear'
}

# Set the root dir
root_dir = os.path.join(save_dir, f'{params["representation_dim"]}', model_name)
if not os.path.isdir(root_dir):
    os.makedirs(root_dir)

loss_path = os.path.join(root_dir, loss_file)
convergence_path = os.path.join(root_dir, convergence_file)

save_autoencoder_params(params, encoder_params, decoder_params, root_dir)

data = HamiltionianDataset(dictionary_path, matrices_path, label_idx=(1, 2))

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])

encoder = Encoder((params['in_channels'], params['N'], params['block_size']), params['representation_dim'], **encoder_params)
decoder = Decoder(params['representation_dim'], (params['in_channels'], params['N'], params['block_size']), **decoder_params)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=params['lr'])
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=params['lr'])

save_data_list(['Epoch', 'Train loss', 'Test loss'], loss_path, mode='w')
tr_loss = test_autoencoder(encoder, decoder, train_loader, device)
te_loss = test_autoencoder(encoder, decoder, test_loader, device)
save_data_list([0, tr_loss, te_loss], loss_path)

for epoch in range(1, params['epochs'] + 1):
    tr_loss = train_autoencoder(encoder, decoder, train_loader, epoch, device, encoder_optimizer, decoder_optimizer)
    te_loss = test_autoencoder(encoder, decoder, test_loader, device)
    save_autoencoder(encoder, decoder, root_dir, epoch)
    save_data_list([epoch, tr_loss, te_loss], loss_path)
plot_convergence(loss_path, convergence_path, read_label=True)
