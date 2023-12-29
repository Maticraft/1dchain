import typing as t

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.data_utils import Hamiltonian
from src.models.gan import Generator
from src.models.utils import get_eigvals, reconstruct_hamiltonian
from src.models.files import DELIMITER


def plot_dim_red_freq_block(
    encoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    file_path: str,
    strategy: str = 'tsne',
    tsne_metric: t.Union[str, t.Callable] = 'euclidean',
    tsne_metric_params: t.Optional[t.Dict[str, t.Any]] = None,
    latent_space_ids: t.Optional[t.List[int]] = None,
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
        z1_list.append(z[:, :z.shape[1] // 2])
        z2_list.append(z[:, z.shape[1] // 2:])
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
):
    encoder_model.to(device)
    encoder_model.eval()

    z_list = []
    y_list = []

    for (x, y), _ in tqdm(test_loader, "Testing dim-red"):
        x = x.to(device)
        z = encoder_model(x).detach().cpu().numpy()
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

    for i in range(len(y.shape[1])):
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
    model: t.Type[Hamiltonian],
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    xaxis: str,
    xparams: t.List[t.Any],
    params: t.Dict[str, t.Any],
    save_path_rec: str,
    save_path_org: t.Optional[str] = None,
    save_path_diff: t.Optional[str] = None,
    **kwargs: t.Dict[str, t.Any],
):
    energies_org = []
    energies_auto = []
    energies_diff = []
    for x in xparams:
        model_params = params.copy()
        if xaxis == 'q_delta_q':
            model_params['q'] = x
            model_params['delta_q'] = x
        else:
            model_params[xaxis] = x
        param_model = model(**model_params)
        H = param_model.get_hamiltonian()

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
    
    eps = 1/num_interval_states
    states = []
    for i in range(len(states_noise) - 1):
        states.append(states_noise[i])
        for j in range(num_interval_states):
            states.append((1-j*eps)*states_noise[i] + j*eps*states_noise[i+1])  
    states.append(states_noise[-1])
    states = torch.stack(states)

    output = generator(states)
    Hs = torch.complex(output[:, 0, :, :], output[:, 1, :, :]).squeeze().detach().cpu().numpy()
    eigvals = [np.linalg.eigvalsh(H) for H in Hs]

    simple_plot(f'{num_states} random states transition', range(len(eigvals)), 'Eigen energy', eigvals, save_path, **kwargs)


def simple_plot(
    xaxis: str,
    xvalues: t.List[t.Any],
    yaxis: str,
    yvalues: t.List[t.Any],
    filename: str,
    **kwargs: t.Dict[str, t.Any]
):
    xnorm = None
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'xnorm' in kwargs and kwargs['xnorm']:
        xvalues = xvalues / kwargs['xnorm']
        if kwargs['xnorm'] == np.pi:
            xnorm = 'π'
        else:
            xnorm = kwargs['xnorm']

    plt.plot(xvalues, yvalues)

    if xnorm:
        plt.xlabel(f'{xaxis}/{xnorm}')
    else:
        plt.xlabel(f'{xaxis}')

    if 'scale' in kwargs:
        plt.yscale(kwargs['scale'])

    plt.ylabel(yaxis)
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
    **reconstruction_kwargs: t.Dict[str, t.Any],
):
    rec_matrix = reconstruct_hamiltonian(matrix, encoder, decoder, device, **reconstruction_kwargs)
    if save_path_org:
        plot_matrix(np.real(matrix), save_path_org.format('_real'))
        plot_matrix(np.imag(matrix), save_path_org.format('_imag'))
    if save_path_rec:
        plot_matrix(np.real(rec_matrix), save_path_rec.format('_real'))
        plot_matrix(np.imag(rec_matrix), save_path_rec.format('_imag'))
    plot_matrix(np.abs(rec_matrix - matrix), save_path_diff, vmin = 1.e-3, vmax = 1, norm = 'log', cmap='YlGnBu')


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
