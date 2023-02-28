import os
import typing as t

import numpy as np
import matplotlib.pyplot as plt

from data_utils import Hamiltonian
from models import Encoder, Decoder, reconstruct_hamiltonian


def count_mzm_states(H: np.ndarray, threshold = 1.e-5):
    eigvals = np.linalg.eigvalsh(H)
    return np.sum(np.abs(eigvals) < threshold)


def majorana_polarization(
    H: np.ndarray,
    threshold: float = 1.e-5,
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
    if site == 'sum':
        return np.sum(list(P_m.values()))
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


def plot_eigvec(H: np.ndarray, component: int, dirpath: str, **kwargs: t.Dict[str, t.Any]):
    if component < 0 or component > 3:
        raise ValueError("Wrong component")
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    eigvals, eigvecs = np.linalg.eigh(H)

    if 'threshold' in kwargs:
        threshold = kwargs['threshold']
        eigvecs = eigvecs[:, np.abs(eigvals) < threshold]
        eigvals = eigvals[np.abs(eigvals) < threshold]

    for i in range(eigvecs.shape[1]):
        real = [np.real(eigvecs[4*site + component, i]) for site in range(eigvecs.shape[0] // 4)]
        imag = [np.imag(eigvecs[4*site + component, i]) for site in range(eigvecs.shape[0] // 4)]

        site_plot(real, dirpath + f'/real_{i}.png', 'Eigenvalue: ' + str(eigvals[i]), 'Real part')
        site_plot(imag, dirpath + f'/imag_{i}.png', 'Eigenvalue: ' + str(eigvals[i]), 'Imaginary part')


def plot_majorana_polarization(
    model: Hamiltonian,
    dirpath: str,
    threshold: float = 1.e-5,
    **kwargs: t.Dict[str, t.Any]
):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    if 'polaxis' in kwargs:
        polaxis = kwargs['polaxis']
    else:
        polaxis = 'total'

    H = model.get_hamiltonian()
    eigvals, eigvecs = np.linalg.eigh(H)
    
    norm_indx = 100
    vec_normed = eigvecs.flatten()[norm_indx] / np.abs(eigvecs.flatten()[norm_indx])

    zm_eigvals = eigvals[np.abs(eigvals) < threshold]
    zm = eigvecs[:, np.abs(eigvals) < threshold] / vec_normed

    P_m_summed = np.zeros(H.shape[0] // 4)
    for i in range(zm.shape[1]):
        P_m = [
            majorana_polarization_site(
                np.expand_dims(zm[4*site:4*(site+1), i], axis=1),
                axis=polaxis,
            ) 
            for site in range(H.shape[0] // 4)
        ]
        P_m_summed += np.array(P_m)

        site_plot(P_m, dirpath + '/polarization_' + str(i) + '.png', 'Eigenvalue: ' + str(zm_eigvals[i]), 'Majorana polarization', **kwargs)

    site_plot(P_m_summed, dirpath + '/polarization_summed.png', 'Summed over eigenvalues', 'Majorana polarization', **kwargs)


def site_plot(values: np.ndarray, filename: str, title: str, ylabel: str, **kwargs: t.Dict[str, t.Any]):
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])

    plt.plot(values)
    plt.title(title)
    plt.xlabel('Site')
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()