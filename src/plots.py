from copy import deepcopy
import os
import typing as t

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.hamiltonian.hamiltonian import IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR, REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR, Hamiltonian
from src.hamiltonian.utils import plot_eigvals_levels, extract_property_strip
from src.hamiltonian.hamiltonian_torch_handlers import BlockConstructor
from src.data_utils import HamiltionianDataset, Denormalize
from src.models.gan import Generator
from src.models.utils import get_eigvals, reconstruct_hamiltonian
from src.models.files import DELIMITER
from src.torch_utils import TorchHamiltonian


def plot_dataset_samples(
    dataset: HamiltionianDataset,
    save_dir: str,
    num_samples: int = 10,
    **kwargs: t.Dict[str, t.Any],
):
    if num_samples > len(dataset):
        raise ValueError(f'Number of samples ({num_samples}) is larger than the dataset size ({len(dataset)})')

    plotted_ids = []
    for i in range(num_samples):
        idx = np.random.randint(len(dataset))
        while idx in plotted_ids:
            idx = np.random.randint(len(dataset))
        (tensor, y), _ = dataset[idx]
        if kwargs.get('label', None) is not None:
            while y.item() != kwargs['label'] or idx in plotted_ids:
                idx = np.random.randint(len(dataset))
                (tensor, y), _ = dataset[idx]
        save_path = os.path.join(save_dir, 'sample_{}.png')

        plot_dataset_sample(tensor, i, save_path, **kwargs)

        if kwargs.get('plot_reconstructed_eigvals', False):
            assert 'encoder' in kwargs and 'decoder' in kwargs, 'Encoder and decoder must be provided for reconstruction'
            plot_reconstructed_sample(tensor, i, save_path, **kwargs)

        plotted_ids.append(idx)


def plot_reconstructed_sample(
    tensor: torch.Tensor,
    sample_idx: int,
    save_path: str,
    encoder: nn.Module,
    decoder: nn.Module,
    **kwargs: t.Dict[str, t.Any],
):
    H = TorchHamiltonian.from_2channel_tensor(tensor)
    H_rec = reconstruct_hamiltonian(H.get_hamiltonian(), encoder, decoder, **kwargs)
    H_rec = TorchHamiltonian(torch.from_numpy(H_rec))
    plot_eigvals_levels(H_rec, save_path.format(f'{sample_idx}_rec_eigvals'), **kwargs)
    plot_matrix(np.real(H_rec.get_hamiltonian()), save_path.format(f'{sample_idx}_rec_matrix_real'), **kwargs)
    plot_matrix(np.imag(H_rec.get_hamiltonian()), save_path.format(f'{sample_idx}_rec_matrix_imag'), **kwargs)


def plot_dataset_sample(tensor: torch.Tensor, sample_idx: int, save_path: str, **kwargs: t.Dict[str, t.Any]):
    H = TorchHamiltonian.from_2channel_tensor(tensor)
    plot_eigvals_levels(H, save_path.format(f'{sample_idx}_eigvals'), **kwargs)
    plot_matrix(np.real(H.get_hamiltonian()), save_path.format(f'{sample_idx}_matrix_real'), **kwargs)
    plot_matrix(np.imag(H.get_hamiltonian()), save_path.format(f'{sample_idx}_matrix_imag'), **kwargs)
        

def plot_dataset_continous_samples(
    dataset: HamiltionianDataset,
    save_dir: str,
    num_samples: int = 10,
    **kwargs: t.Dict[str, t.Any],
):
    if num_samples > len(dataset):
        raise ValueError(f'Number of samples ({num_samples}) is larger than the dataset size ({len(dataset)})')

    plotted_ids = []
    num_hamiltonians = kwargs.get('num_hamiltonians', 2)
    num_steps = kwargs.get('num_steps', 10)
    eps = np.linspace(0, 1, num_steps)
    for i in tqdm(range(num_samples), desc='Plotting samples'):
        hamiltonians = []
        for _ in range(num_hamiltonians):
            idx = np.random.randint(len(dataset))
            while idx in plotted_ids:
                idx = np.random.randint(len(dataset))
            (tensor, y), _ = dataset[idx]
            if kwargs.get('label', None) is not None:
                while y.item() != kwargs['label'] or idx in plotted_ids:
                    idx = np.random.randint(len(dataset))
                    (tensor, y), _ = dataset[idx]
            H = TorchHamiltonian.from_2channel_tensor(tensor)
            hamiltonians.append(H)

        total_eigvals = []
        for j in range(num_hamiltonians - 1):
            H1 = hamiltonians[j]
            H2 = hamiltonians[j + 1]
            for eps_k in eps:
                H_k = (1 - eps_k) * H1.get_hamiltonian() + eps_k * H2.get_hamiltonian()
                eigvals = np.linalg.eigvalsh(H_k)
                total_eigvals.append(eigvals)
            
        save_path = os.path.join(save_dir, f'continous_egivals_spectre_{i}.png')
        simple_plot(f'{num_hamiltonians} hamiltonians transition', range(len(total_eigvals)), 'Eigen energy', total_eigvals, save_path, **kwargs)


def plot_dim_red_freq_block(
    encoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    file_path: str,
    strategy: str = 'tsne',
    tsne_metric: t.Union[str, t.Callable] = 'euclidean',
    tsne_metric_params: t.Optional[t.Dict[str, t.Any]] = None,
    latent_space_ids: t.Optional[t.List[int]] = None,
    num_freq_features: int = 50,
):
    encoder_model.to(device)
    encoder_model.eval()

    z1_list = []
    z2_list = []
    y_list = []

    for (x, y), _ in tqdm(test_loader, "Testing dim-red"):
        x = x.to(device)
        z = encoder_model(x).detach().cpu().numpy()
        if latent_space_ids is not None:
            z = z[:, latent_space_ids]
        z1_list.append(z[:, :num_freq_features])
        z2_list.append(z[:, num_freq_features:])
        y_list.append(y.detach().cpu().numpy())

    if strategy == 'tsne':
        plot_tsne(file_path.format('_freq'), z1_list, y_list, metric=tsne_metric, metric_params=tsne_metric_params)
        plot_tsne(file_path.format('_block'), z2_list, y_list, metric=tsne_metric, metric_params=tsne_metric_params)
    elif strategy == 'pca':
        plot_pca(file_path.format('_freq'), z1_list, y_list)
        plot_pca(file_path.format('_block'), z2_list, y_list)
    else:
        raise ValueError(f'Unknown strategy: {strategy}')


def plot_dim_red_full_space(
    encoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    file_path: str,
    strategy: str = 'tsne',
    tsne_metric: t.Union[str, t.Callable] = 'euclidean',
    tsne_metric_params: t.Optional[t.Dict[str, t.Any]] = None,
    latent_space_ids: t.Optional[t.List[int]] = None,
    predictor: t.Optional[nn.Module] = None,
):
    encoder_model.to(device)
    encoder_model.eval()
    if predictor is not None:
        predictor.to(device)
        predictor.eval()

    z_list = []
    y_list = []

    for (x, y), _ in tqdm(test_loader, "Testing dim-red"):
        x = x.to(device)
        z = encoder_model(x)
        if predictor is not None:
            y = predictor(z).squeeze()
        z = z.detach().cpu().numpy()
        if latent_space_ids is not None:
            z = z[:, latent_space_ids]
        z_list.append(z)
        y_list.append(y.detach().cpu().numpy())

    if strategy == 'tsne':
        plot_tsne(file_path, z_list, y_list, metric=tsne_metric, metric_params=tsne_metric_params)
    elif strategy == 'pca':
        plot_pca(file_path, z_list, y_list)
    else:
        raise ValueError(f'Unknown strategy: {strategy}')


def plot_tsne(
    file_path: str,
    z_list: t.List[np.ndarray],
    y_list: t.List[np.ndarray],
    metric=t.Union[str, t.Callable],
    metric_params: t.Optional[t.Dict[str, t.Any]] = None,
):
    z = np.concatenate(z_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    tsne = TSNE(n_components=2, random_state=0, metric=metric, metric_params=metric_params)
    z_tsne = tsne.fit_transform(z)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    for i in range(y.shape[1]):
        file_path_rep = file_path.replace('.png', f'_{i}.png')
        plt.figure(figsize=(10, 10))
        plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y[:, i])
        plt.colorbar()
        plt.axis('off')
        plt.savefig(file_path_rep)
        plt.close()


def plot_pca(file_path: str, z_list: t.List[np.ndarray], y_list: t.List[np.ndarray]):
    z = np.concatenate(z_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z)
    print(pca.explained_variance_ratio_)

    if len(y.shape) == 1:
        y = y.reshape(-1, 1)

    for i in range(y.shape[1]):
        file_path_rep = file_path.replace('.png', f'_{i}.png')
        plt.figure(figsize=(10, 10))
        plt.scatter(z_pca[:, 0], z_pca[:, 1], c=y[:, i])
        plt.colorbar()
        plt.savefig(file_path_rep)
        plt.close()


def plot_convergence(results_path: str, save_path: str, read_label: bool = False):
    if read_label:
        with open(results_path) as f:
            labels = f.readline()
            data = f.readlines()
        labels = labels.split(DELIMITER)
        data = [[float(x) for x in row.split(DELIMITER)] for row in data]
        data = np.array(data)
    else:
        with open(results_path) as f:
            data = f.readlines()
        data = [[float(x) for x in row.split(DELIMITER)] for row in data]
        data = np.array(data)
        labels = [f'{i}' for i in range(len(data[0, :]))]

    for i in range(1, len(data[0, :])):
        plt.plot(data[:, 0], data[:, i], label = labels[i])
    plt.xlabel(labels[0])
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_test_eigvals(
    model: Hamiltonian,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    xaxis: str,
    xparams: t.List[t.Any],
    save_path_rec: str,
    save_path_org: t.Optional[str] = None,
    save_path_diff: t.Optional[str] = None,
    **kwargs: t.Dict[str, t.Any],
):
    energies_org = []
    energies_auto = []
    energies_diff = []
    for x in xparams:
        new_model = deepcopy(model)
        if xaxis == 'q_delta_q':
            new_model.set_parameter('q', x)
            new_model.set_parameter('delta_q', x)
        else:
            new_model.set_parameter(xaxis, x)
        H = new_model.get_hamiltonian()

        auto_eigvals, eigvals = get_eigvals(H, encoder, decoder, return_ref_eigvals=True, **kwargs)
        eigvals_diff = np.mean(np.abs(eigvals - auto_eigvals))
        energies_diff.append(eigvals_diff)
        energies_org.append(eigvals)
        energies_auto.append(auto_eigvals)

    simple_plot(xaxis, xparams, 'Autoencoder energy', energies_auto, save_path_rec, **kwargs)
    if save_path_org:
        simple_plot(xaxis, xparams, 'Energy', energies_org, save_path_org, **kwargs)
    if save_path_diff:
        kwargs['ylim'] = (1.e-4, 1.)
        kwargs['scale'] = 'log'
        simple_plot(xaxis, xparams, 'Energy difference', energies_diff, save_path_diff, **kwargs)


def plot_generator_eigvals(
    generator: Generator,
    num_states: int,
    save_path: str,
    noise_type: str = 'hybrid',
    num_interval_states: int = 100,
    **kwargs: t.Dict[str, t.Any],
):
    generator = generator.cpu()
    generator.eval()
    
    states_noise = generator.get_noise(num_states, torch.device('cpu'), noise_type, **kwargs)
    if 'real_sample' in kwargs:
        states_noise = (1 - kwargs['noise_strength'])*kwargs['real_sample'] + kwargs['noise_strength']*states_noise

    eps = 1/num_interval_states
    states = []
    for i in range(len(states_noise) - 1):
        states.append(states_noise[i])
        for j in range(num_interval_states):
            states.append((1-j*eps)*states_noise[i] + j*eps*states_noise[i+1])  
    states.append(states_noise[-1])
    states = torch.stack(states)

    output = generator(states)
    if 'normalization_mean' in kwargs and 'normalization_std' in kwargs:
        denormalization = Denormalize(kwargs['normalization_mean'], kwargs['normalization_std'])
        output = denormalization(output)
    
    if kwargs.get('plot_properties_change', False):
        for property_name, property_config in kwargs.get('properties', {'potential': {'interaction_level': 0, 'part': 'all'}}).items():
            if property_config['part'] == 'all' or property_config['part'] == 'real':
                real_avg_values = _collect_avg_property_values_for_hamiltonian_tensor(output, property_name, property_config['interaction_level'], 'real')
                simple_plot(f'{num_states} random states transition', range(len(real_avg_values)), f'{property_name} real', real_avg_values, save_path.replace('.png', f'_{property_name}_real.png'), **kwargs)
            if property_config['part'] == 'all' or property_config['part'] == 'imag':
                imag_avg_values = _collect_avg_property_values_for_hamiltonian_tensor(output, property_name, property_config['interaction_level'], 'imag')
                simple_plot(f'{num_states} random states transition', range(len(imag_avg_values)), f'{property_name} imag', imag_avg_values, save_path.replace('.png', f'_{property_name}_imag.png'), **kwargs)

    Hs = torch.complex(output[:, 0, :, :], output[:, 1, :, :]).squeeze().detach().cpu().numpy()
    eigvals = [np.linalg.eigvalsh(H) for H in Hs]

    simple_plot(f'{num_states} random states transition', range(len(eigvals)), 'Eigen energy', eigvals, save_path, **kwargs)


def _collect_avg_property_values_for_hamiltonian_tensor(
    hamiltonian_tensor: torch.Tensor,
    property_name: str,
    interaction_level: int,
    part: str,
):
    avg_values = []
    for h in hamiltonian_tensor:
        hamiltonian = TorchHamiltonian.from_2channel_tensor(h)
        strip = extract_property_strip(hamiltonian, property_name, part=part, interaction_level=interaction_level)
        avg_values.append(np.mean(strip))
    return avg_values


def plot_generator_noisy_sample_eigvals(
    generator: Generator,
    save_path: str,
    real_sample: torch.Tensor,
    noise_type: str = 'hybrid',
    num_interval_states: int = 100,
    eps = 1/100,
    **kwargs: t.Dict[str, t.Any],
):
    generator = generator.cpu()
    generator.eval()
    
    states_noise = generator.get_noise(num_interval_states, torch.device('cpu'), noise_type, **kwargs)    
    states = torch.cat([(1-eps)*real_sample + eps*noise for noise in states_noise])

    output = generator(states)
    if 'normalization_mean' in kwargs and 'normalization_std' in kwargs:
        denormalization = Denormalize(kwargs['normalization_mean'], kwargs['normalization_std'])
        output = denormalization(output)
    
    Hs = torch.complex(output[:, 0, :, :], output[:, 1, :, :]).squeeze().detach().cpu().numpy()
    eigvals = [np.linalg.eigvalsh(H) for H in Hs]

    simple_plot(f'{num_interval_states} noise samples transition', range(len(eigvals)), 'Eigen energy', eigvals, save_path, **kwargs)


def plot_generator_sample_eigvals_varying_property(
    save_path: str,
    real_sample: torch.Tensor,
    property_name: str,
    part: str = 'both',
    num_interval_states: int = 100,
    offset: int = 0,
    **kwargs: t.Dict[str, t.Any],
):
    real_property_block_names = []
    imag_property_block_names = []
    init_block_params = torch.ones((1, 1, real_sample.shape[-1] // 4))
    if part == 'real' or part == 'both':
        real_property_block_names.append(REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR[property_name])
    if part == 'imag' or part == 'both':
        imag_property_block_names.append(IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR[property_name])
    block_matrix = torch.abs(BlockConstructor.generate_block_matrix(init_block_params, init_block_params, real_property_block_names, imag_property_block_names, offset))
    weights = np.linspace(1.e-5, 1., num_interval_states - 1)
    weight_matrices = torch.cat([block_matrix*weights[i] for i in range(num_interval_states - 1)], dim=0)
    weight_matrices_0 = torch.where(block_matrix == 0, torch.ones_like(block_matrix), torch.zeros_like(block_matrix))
    weight_matrices = torch.cat((weight_matrices_0, torch.where(weight_matrices == 0, torch.ones_like(weight_matrices), weight_matrices)))
    weighted_matrices = real_sample.unsqueeze(0) * weight_matrices
    Hs = torch.complex(weighted_matrices[:, 0, :, :], weighted_matrices[:, 1, :, :]).squeeze().detach().cpu().numpy()
    eigvals = [np.linalg.eigvalsh(H) for H in Hs]

    # kwargs['xscale'] = 'log'
    simple_plot(f'varying {property_name} amplitude', np.append(weights, 0), 'Eigen energy', eigvals, save_path, **kwargs)


def plot_generator_sample_eigvals_increasing_property(
    save_path: str,
    real_sample: torch.Tensor,
    property_name: str,
    property_value_range: t.Tuple[float, float],
    part: str = 'both',
    num_interval_states: int = 100,
    offset: int = 0,
    should_replace_original_property: bool = False, 
    **kwargs: t.Dict[str, t.Any],
):
    real_property_block_names = []
    imag_property_block_names = []
    init_block_params = torch.ones((1, 1, real_sample.shape[-1] // 4))
    if part == 'real' or part == 'both':
        real_property_block_names.append(REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR[property_name])
    if part == 'imag' or part == 'both':
        imag_property_block_names.append(IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR[property_name])
    block_matrix = BlockConstructor.generate_block_matrix(init_block_params, init_block_params, real_property_block_names, imag_property_block_names, offset)
    property_values = np.linspace(property_value_range[0], property_value_range[1], num_interval_states)
    property_matrices = torch.cat([block_matrix*property_values[i] for i in range(num_interval_states)], dim=0)
    modified_sample = real_sample.unsqueeze(0)
    if should_replace_original_property:
        modified_sample = torch.where(block_matrix == 0, modified_sample, torch.zeros_like(modified_sample))
    weighted_matrices = modified_sample + property_matrices
    Hs = torch.complex(weighted_matrices[:, 0, :, :], weighted_matrices[:, 1, :, :]).squeeze().detach().cpu().numpy()
    eigvals = [np.linalg.eigvalsh(H) for H in Hs]

    # kwargs['xscale'] = 'log'
    simple_plot(f'{property_name} ', property_values, 'Eigen energy', eigvals, save_path, **kwargs)



def simple_plot(
    xaxis: str,
    xvalues: t.List[t.Any],
    yaxis: str,
    yvalues: t.List[t.Any],
    filename: str,
    **kwargs: t.Dict[str, t.Any]
):
    xnorm = None
    ynorm = None
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'xnorm' in kwargs and kwargs['xnorm']:
        xvalues = np.array(xvalues) / kwargs['xnorm']
        if kwargs['xnorm'] == np.pi:
            xnorm = 'π'
        else:
            xnorm = kwargs['xnorm']
    if 'ynorm' in kwargs:
        yvalues = np.array(yvalues) / kwargs['ynorm']
        if kwargs['ynorm'] == np.pi:
            ynorm = 'π'
        else:
            ynorm = kwargs['ynorm']

    plt.plot(xvalues, yvalues)

    if xnorm:
        plt.xlabel(f'{xaxis}/{xnorm}')
    else:
        plt.xlabel(f'{xaxis}')
    if ynorm:
        plt.ylabel(f'{yaxis}/{ynorm}')
    else:
        plt.ylabel(yaxis)

    if 'scale' in kwargs:
        plt.yscale(kwargs['scale'])
    if 'xscale' in kwargs:
        plt.xscale(kwargs['xscale'])

    plt.savefig(filename)
    plt.close()


def plot_test_matrices(
    matrix: np.ndarray,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    save_path_diff: str,
    save_path_rec: t.Optional[str] = None,
    save_path_org: t.Optional[str] = None,
    device: torch.device = torch.device('cpu'),
    vscale: t.Optional[float] = None,
    **reconstruction_kwargs: t.Dict[str, t.Any],
):
    if vscale is None:
        vscale = 1.
    rec_matrix = reconstruct_hamiltonian(matrix, encoder, decoder, device, **reconstruction_kwargs)
    if save_path_org:
        plot_matrix(np.real(matrix), save_path_org.format('_real'), vmin = -vscale, vmax = vscale)
        plot_matrix(np.imag(matrix), save_path_org.format('_imag'), vmin = -vscale, vmax = vscale)
    if save_path_rec:
        plot_matrix(np.real(rec_matrix), save_path_rec.format('_real'), vmin = -vscale, vmax = vscale)
        plot_matrix(np.imag(rec_matrix), save_path_rec.format('_imag'), vmin = -vscale, vmax = vscale)
    plot_matrix(np.abs(rec_matrix - matrix), save_path_diff, vmin = vscale*1.e-3, vmax = vscale*1, norm = 'log', cmap='YlGnBu')


def plot_matrix(matrix: np.ndarray, filepath: str, **kwargs: t.Dict[str, t.Any]):
    vmin = kwargs.get('vmin', -0.5)        
    vmax = kwargs.get('vmax', 0.5)
    norm = kwargs.get('norm', None)
    fig = plt.figure()
    if 'cmap' in kwargs:
        cmap = kwargs['cmap']
    else:
        cmap = 'PuOr'
    im = plt.imshow(matrix, cmap=cmap, vmin = vmin, vmax = vmax, norm=norm)
    cbar = fig.colorbar(im, shrink=0.9)
    cbar.ax.tick_params(labelsize=35)
    plt.savefig(filepath, dpi=560)
    plt.close()


def plot_latent_space_distribution(
    latent_space_distribution: t.Tuple[torch.Tensor, torch.Tensor],
    save_path: str,
):
    mean, std = latent_space_distribution
    plt.errorbar(
        x = np.arange(len(mean)),
        y = mean.detach().cpu().numpy(),
        yerr = std.detach().cpu().numpy(),
        fmt = 'o',
        capsize = 5,
    )
    plt.xlabel('Latent space index')
    plt.ylabel('Mean and standard deviation')
    plt.savefig(save_path)
    plt.close()
