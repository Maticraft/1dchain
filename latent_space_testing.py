import os

from torch.utils.data import DataLoader
import torch


from data_utils import HamiltionianDataset

from models_utils import calculate_latent_space_distribution
from models_files import load_variational_positional_autoencoder, load_positional_autoencoder, save_latent_distribution
from models_plots import plot_full_space_tsne, plot_tsne_freq_block, plot_latent_space_distribution

model_dir = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPG/100/classifier_bal_twice_pretrained_positional_autoencoder_fft_tf'
epoch = 4
batch_size = 64

# Paths
data_path = './data/spin_ladder/70_2_RedDistSimplePeriodicPG'

test_dir_name = 'tests_ep{}'
tsne_file_name = 'tsne_{}.png'
latent_space_plot_name = 'latent_space.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
encoder, decoder = load_positional_autoencoder(model_dir, epoch)

data = HamiltionianDataset(data_path, data_limit=1000, label_idx=(3, 4), eig_decomposition=False, format='csr')
test_loader = DataLoader(data, batch_size)

dir_path = os.path.join(model_dir, test_dir_name.format(epoch))
if not os.path.isdir(dir_path):
    os.makedirs(dir_path)

tsne_path = os.path.join(dir_path, tsne_file_name)
latent_space_plot_path = os.path.join(dir_path, latent_space_plot_name)

latent_space_distribution = calculate_latent_space_distribution(encoder, test_loader, device)
save_latent_distribution(latent_space_distribution, dir_path)
plot_latent_space_distribution(latent_space_distribution, latent_space_plot_path)
plot_full_space_tsne(encoder, test_loader, device, tsne_path.format('all'))
plot_tsne_freq_block(encoder, test_loader, device, tsne_path)