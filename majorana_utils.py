import typing as t

import numpy as np
import matplotlib.pyplot as plt

from data_utils import Hamiltonian
from models import Encoder, Decoder, reconstruct_hamiltonian


def count_mzm_states(H: np.ndarray, threshold = 1e-5):
    eigvals = np.linalg.eigvalsh(H)
    return np.sum(np.abs(eigvals) < threshold)


def majorana_polarization(
    H: np.ndarray,
    threshold: float = 1e-5,
    axis: str = 'total',
    site: t.Optional[t.Union[int, str]] = 'avg'
):
    eigvals, eigvecs = np.linalg.eigh(H)
    zm = eigvecs[:, np.abs(eigvals) < threshold]
    if zm.shape[1] == 0:
        if site == 'all':
            return {'all': 0}
        return 0.

    if type(site) == int:
        zm_site = zm[4*site:4*(site+1), :]
        return majorana_polarization_site(zm_site, axis=axis)

    P_m = {}
    for i in range(zm.shape[0] // 4):
        zm_site_i = zm[4*i:4*(i+1), :]
        P_m[i] = majorana_polarization_site(zm_site_i, axis=axis)
        
    if site == 'avg':
        return np.mean(list(P_m.values()))
    elif site == 'all':
        return P_m
    else:
        raise ValueError('site must be one of "avg", "all", or an integer')


def majorana_polarization_site(zero_mode: np.ndarray, axis: str = 'total'):
    if axis == 'total':
        return 2*np.sum(np.abs(zero_mode[1, :] * zero_mode[3, :].conj() - zero_mode[0, :] * zero_mode[2, :].conj()))
    if axis == 'x':
        return 2*np.sum(np.real(zero_mode[1, :] * zero_mode[3, :].conj() - zero_mode[0, :] * zero_mode[2, :].conj()))
    if axis == 'y':
        return 2*np.sum(np.imag(zero_mode[1, :] * zero_mode[3, :].conj() - zero_mode[0, :] * zero_mode[2, :].conj()))


def plot_eigvals(model: Hamiltonian, xaxis: str, xparams: t.List[t.Any], params: t.Dict[str, t.Any], filename: str, **kwargs: t.Dict[str, t.Any]):
    energies = []
    for x in xparams:
        model_params = params.copy()
        model_params[xaxis] = x
        ladder = model(**model_params)
        energies.append(np.linalg.eigvalsh(ladder.get_hamiltonian()))

    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'xnorm' in kwargs:
        xparams = xparams / kwargs['xnorm']

    plt.plot(xparams, energies)
    plt.xlabel('q/π')
    plt.ylabel('Energy')
    plt.savefig(filename)
    plt.close()


def plot_autoencoder_eigvals(
    model: Hamiltonian,
    encoder: Encoder,
    decoder: Decoder,
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

    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])
    if 'xnorm' in kwargs:
        xparams = xparams / kwargs['xnorm']

    plt.plot(xparams, energies)
    plt.xlabel('q/π')
    plt.ylabel('Energy')
    plt.savefig(filename)
    plt.close()


def plot_eigvec(H: np.ndarray, component: int, filename: str, **kwargs: t.Dict[str, t.Any]):
    if component < 0 or component > 3:
        raise ValueError("Wrong component")
    eigvals, eigvecs = np.linalg.eigh(H)

    density = []
    for i in range(eigvecs.shape[0] // 4):
        density.append(np.abs(eigvecs[4*i + component, np.abs(eigvals) < 1.e-5]))

    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])

    chart = plt.plot(density)
    plt.legend(chart, ('C+^', 'C+v', 'C^', 'Cv'))
    plt.xlabel('Site')
    plt.ylabel('Probability density')
    plt.savefig(filename)
    plt.close()