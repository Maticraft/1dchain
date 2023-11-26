import os

from seaborn import heatmap
from torch.utils.data import DataLoader
import torch


from data_utils import HamiltionianDataset, HamiltionianParamsDataset

from models_utils import calculate_latent_space_distribution, calculate_classifier_distance
from models_files import load_classifier, load_variational_positional_autoencoder, load_positional_autoencoder, load_general_params, save_latent_distribution, save_covariance_matrix
from models_plots import plot_dim_red_full_space, plot_dim_red_freq_block, plot_latent_space_distribution

model_dir = './autoencoder/spin_ladder/70_2_RedDistSimplePeriodicPG/100/classifier_bal_twice_pretrained_positional_autoencoder_fft_tf'
epoch = 4
batch_size = 128

# Paths
data_path = './data/spin_ladder/70_2_RedDistSimplePeriodicPG'

test_dir_name = 'tests_majoranas_ep{}'
dim_red_plot_file_name = 'tsne_class_{}.png'
latent_space_plot_name = 'latent_space.png'
correlation_matrix_name = 'covariance_matrix.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
params = load_general_params(model_dir)
encoder, decoder = load_positional_autoencoder(model_dir, epoch)
classifier = load_classifier(model_dir, epoch, input_dim = params['representation_dim'])

data = HamiltionianDataset(data_path, data_limit=1000, label_idx=(3, 4), eig_decomposition=False, format='csr')
# data = HamiltionianParamsDataset(data_path, data_limit=1000, label_key=['potential', 'increase_potential_at_edges'], format='csr')
test_loader = DataLoader(data, batch_size)

dir_path = os.path.join(model_dir, test_dir_name.format(epoch))
if not os.path.isdir(dir_path):
    os.makedirs(dir_path)

dim_red_plot_path = os.path.join(dir_path, dim_red_plot_file_name)
latent_space_plot_path = os.path.join(dir_path, latent_space_plot_name)
cov_matrix_path = os.path.join(dir_path, correlation_matrix_name)

# latent_space_distribution = calculate_latent_space_distribution(encoder, test_loader, device, label=1)
# mean, std, covariance_matrix = latent_space_distribution

# save_latent_distribution((mean, std), dir_path)
# plot_latent_space_distribution((mean, std), latent_space_plot_path)
plot_dim_red_full_space(encoder, test_loader, device, dim_red_plot_path.format('all'), strategy='tsne', tsne_metric=calculate_classifier_distance, tsne_metric_params={'model': classifier})
plot_dim_red_freq_block(encoder, test_loader, device, dim_red_plot_path, strategy='tsne')
# save_covariance_matrix(covariance_matrix, dir_path)
# cov_matrix_plot = heatmap(covariance_matrix.detach().cpu().numpy(), annot=False)
# cov_matrix_plot.get_figure().savefig(cov_matrix_path)
