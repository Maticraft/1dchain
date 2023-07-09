import typing as t

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import TSNE

from data_utils import Hamiltonian
from models_utils import get_eigvals, reconstruct_hamiltonian
from models_files import DELIMITER


def plot_tsne_freq_block(
    encoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    file_path: str,
):
    encoder_model.to(device)
    encoder_model.eval()

    z1_list = []
    z2_list = []
    y_list = []

    for (x, y), _ in tqdm(test_loader, "Testing t-SNE"):
        x = x.to(device)
        z = encoder_model(x).detach().cpu().numpy()
        z1_list.append(z[:, :z.shape[1] // 2])
        z2_list.append(z[:, z.shape[1] // 2:])
        y_list.append(y.detach().cpu().numpy())

    plot_tsne(file_path.format('_freq'), z1_list, y_list)
    plot_tsne(file_path.format('_block'), z2_list, y_list)


def plot_full_space_tsne(
    encoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    file_path: str,
):
    encoder_model.to(device)
    encoder_model.eval()

    z_list = []
    y_list = []

    for (x, y), _ in tqdm(test_loader, "Testing t-SNE"):
        x = x.to(device)
        z = encoder_model(x).detach().cpu().numpy()
        z_list.append(z)
        y_list.append(y.detach().cpu().numpy())

    plot_tsne(file_path, z_list, y_list)


def plot_tsne(file_path, z_list, y_list):
    z = np.concatenate(z_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    z_tsne = tsne.fit_transform(z)

    plt.figure(figsize=(10, 10))
    plt.scatter(z_tsne[:, 0], z_tsne[:, 1], c=y, cmap='bwr')
    plt.axis('off')
    plt.savefig(file_path)


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

        auto_eigvals, eigvals = get_eigvals(H, encoder, decoder, kwargs.get('device', None), return_ref_eigvals=True)
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
):
    rec_matrix = reconstruct_hamiltonian(matrix, encoder, decoder, device)
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
