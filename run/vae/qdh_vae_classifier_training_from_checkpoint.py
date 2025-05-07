import os
import pickle

import numpy as np
from torch.utils.data import random_split, DataLoader
import torch
from src.data.datasets import HamiltionianDataset
from src.hamiltonian.units import AtomicUnits
from src.models.autoencoder import test_autoencoder, train_autoencoder
from src.models.classifier import Classifier, test_encoder_with_classifier

from src.data.utils import calculate_mean_and_std
from src.hamiltonian.quantum_dots_chain import QuantumDotsHamiltonian, QuantumDotsHamiltonianParameters, DefaultParameters, MZM_THRESHOLD
from src.models.distribution_preserving_autoencoder import DistributionPreservingHamiltonianGenerator, VariationalDistributionPreservingEncoder, train_vae
from src.models.classifier import train_encoder_with_classifier
from src.models.files import save_autoencoder_params, save_autoencoder, save_data_list, load_autoencoder_params, load_ae_model, save_model
from src.plots import plot_convergence, plot_test_matrices, plot_test_eigvals
from src.models.positional_autoencoder import PositionalDecoder

# Pretrained model
pretrained_model_dir = './vae/quantum_dots/7dots2levels_large/100/distribution_preserving_autoencoder_no_kl_debug'
epoch = 28

# Paths
data_path = './data/quantum_dots/7dots2levels_large_balanced'
data_mean_std_path = f'{data_path}/mean_std.pkl'
save_dir = './vae/quantum_dots/7dots2levels_large_balanced'
loss_file = 'loss.txt'
convergence_file = 'convergence.png'

# Reference eigvals plot params
eigvals_sub_dir = 'eigvals'
eigvals_plot_name = 'eigvals_spectre_autoencoder{}.png'
eigvals_org_plot_name = 'eigvals_spectre_original.png'
x_axis = 'mu'
x_values = np.linspace(-1.5/AtomicUnits.Eh, .5/AtomicUnits.Eh, 100)
xnorm=1/AtomicUnits.Eh
ynorm=1/AtomicUnits.Eh
ylim = (-1., 1.)
vscale = 1/AtomicUnits.Eh

# Reference hamiltonian params
hamiltonian_sub_dir = 'hamiltonian'
hamiltonian_plot_name = 'hamiltonian_autoencoder{}.png'
hamiltonian_diff_plot_name = 'hamiltonian_diff{}.png'
hamiltonian_org_plot_name = 'hamiltonian_original{}.png'

default_params = DefaultParameters()
parameters = QuantumDotsHamiltonianParameters(no_dots=7, no_levels=2, default_parameters=default_params)
parameters.set_random_parameters_const()
test_hamiltonian = QuantumDotsHamiltonian(parameters)

# New model name
model_name = 'classifier_pretrained_distribution_preserving_autoencoder'

# Load model
encoder, decoder = load_ae_model(pretrained_model_dir, epoch, VariationalDistributionPreservingEncoder, DistributionPreservingHamiltonianGenerator)
params, encoder_params, decoder_params = load_autoencoder_params(pretrained_model_dir, VariationalDistributionPreservingEncoder, PositionalDecoder)

# Modify params
params['learning_rate'] = 1.e-5
params['epochs'] = 40

classifier = Classifier(params['representation_dim'], 1)

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    with open(data_mean_std_path, 'rb') as f:
        mean, std = pickle.load(f)
except:
    data = HamiltionianDataset(data_path, label_idx=(3, 4), format='csr', threshold=MZM_THRESHOLD)
    data_loader = DataLoader(data, params['batch_size'])
    mean, std = calculate_mean_and_std(data_loader, device=device)
    with open(data_mean_std_path, 'wb') as f:
        pickle.dump((mean, std), f)

data = HamiltionianDataset(data_path, label_idx=(3, 4), eig_decomposition=params['eigenstates_loss'], format='csr', normalization_mean=mean, normalization_std=std, threshold=MZM_THRESHOLD)

train_size = int(0.99*len(data))
test_size = len(data) - train_size

train_data, test_data = random_split(data, [train_size, test_size])
train_loader = DataLoader(train_data, params['batch_size'])
test_loader = DataLoader(test_data, params['batch_size'])


encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=params['lr'])
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=params['lr'])
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=params['lr'])
encoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min')
decoder_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min')
classifier_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(classifier_optimizer, 'min')

save_data_list(['Epoch', 'Train_classifier_loss', 'Train_ae_loss', 'Test_classifier_loss', 'Test_classifier_acc', 'Test_ae_loss', 'Test_edge_loss', 'Test_eigenstates_loss', 'Te_diag_loss'], loss_path, mode='w')

for epoch in range(1, params['epochs'] + 1):
    tr_class_loss, tr_ae_loss = train_encoder_with_classifier(encoder, decoder, classifier, train_loader, epoch, device, encoder_optimizer, decoder_optimizer, classifier_optimizer)

    te_class_loss, te_acc, te_cm, te_reg = test_encoder_with_classifier(encoder, classifier, test_loader, device)
    te_loss, te_edge_loss, te_eig_loss, te_eig_loss, te_diag_loss, te_det_loss = test_autoencoder(encoder, decoder, test_loader, device, edge_loss=params['edge_loss'], eigenstates_loss=params['eigenstates_loss'], diag_loss=params['diag_loss'])
    classifier_scheduler.step(te_class_loss)
    encoder_scheduler.step(te_loss)
    decoder_scheduler.step(te_loss)

    save_autoencoder(encoder, decoder, root_dir, epoch)
    save_model(classifier, root_dir, epoch)
    save_data_list([epoch, tr_class_loss, tr_ae_loss, te_class_loss, te_acc, te_loss, te_edge_loss, te_eig_loss, te_diag_loss], loss_path)

    eigvals_path = os.path.join(eigvals_sub_path, eigvals_plot_name.format(f'_ep{epoch}'))
    eigvals_org_path = os.path.join(eigvals_sub_path, eigvals_org_plot_name)
    plot_test_eigvals(test_hamiltonian, encoder, decoder, x_axis, x_values, save_path_rec=eigvals_path, save_path_org=eigvals_org_path, device=device, xnorm=xnorm, ynorm=ynorm, ylim=ylim, normalization_mean=mean, normalization_std=std)
    ham_auto_path = os.path.join(ham_sub_path, hamiltonian_plot_name.format(f'_ep{epoch}' + '{}'))
    ham_diff_path = os.path.join(ham_sub_path, hamiltonian_diff_plot_name.format(f'_ep{epoch}'))
    ham_org_path = os.path.join(ham_sub_path, hamiltonian_org_plot_name)
    plot_test_matrices(test_hamiltonian.get_hamiltonian(), encoder, decoder, save_path_rec=ham_auto_path, save_path_diff=ham_diff_path, save_path_org=ham_org_path, device=device, normalization_mean=mean, normalization_std=std, vscale=vscale)
   
plot_convergence(loss_path, convergence_path, read_label=True)
