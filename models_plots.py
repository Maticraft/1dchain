import typing as t

import matplotlib.pyplot as plt
import numpy as np
import torch

from data_utils import Hamiltonian
from models_utils import reconstruct_hamiltonian
from models_files import DELIMITER


def plot_autoencoder_eigvals(
    model: Hamiltonian,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    xaxis: str,
    xparams: t.List[t.Any],
    params: t.Dict[str, t.Any],
    filename: str,
    **kwargs: t.Dict[str, t.Any]
):
    energies = []
    for x in xparams:
        model_params = params.copy()
        model_params[xaxis] = x
        param_model = model(**model_params)

        H = param_model.get_hamiltonian()

        if 'device' in kwargs:
            H_rec = reconstruct_hamiltonian(H, encoder, decoder, kwargs['device'])
        else:
            H_rec = reconstruct_hamiltonian(H, encoder, decoder)

        energies.append(np.linalg.eigvalsh(H_rec))

    xnorm = None
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'xnorm' in kwargs:
        xparams = xparams / kwargs['xnorm']
        if kwargs['xnorm'] == np.pi:
            xnorm = 'π'
        else:
            xnorm = kwargs['xnorm']

    plt.plot(xparams, energies)

    if xnorm:
        plt.xlabel(f'{xaxis}/{xnorm}')
    else:
        plt.xlabel(f'{xaxis}')
    plt.ylabel('Energy')
    plt.savefig(filename)
    plt.close()


def plot_convergence(results_path: str, save_path: str, read_label: bool = False):
    if read_label:
        skip_rows = 1
        with open(results_path) as f:
            labels = f.readline()
            data = f.readlines()
        labels = labels.split(DELIMITER)
        data = [[float(x) for x in row.split(DELIMITER)] for row in data]
        data = np.array(data)
    else:
        skip_rows = 0
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


def plot_matrix(matrix: np.ndarray, filepath: str, **kwargs: t.Dict[str, t.Any]):
    vmin = kwargs.get('vmin', -0.5)        
    vmax = kwargs.get('vmax', 0.5)
    norm = kwargs.get('norm', None)
    fig = plt.figure()
    im = plt.imshow(matrix, cmap='PuOr', vmin = vmin, vmax = vmax, norm=norm)
    cbar = fig.colorbar(im, shrink=0.9)
    cbar.ax.tick_params(labelsize=35)
    plt.savefig(filepath)
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
    plot_matrix(np.abs(rec_matrix - matrix), save_path_diff, vmin = 1.e-7, vmax = 1, norm = 'log')