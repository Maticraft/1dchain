import typing as t

import numpy as np
from data_utils import Hamiltonian
import matplotlib.pyplot as plt


def count_mzm_states(H: np.ndarray):
    eigvals = np.linalg.eigvalsh(H)
    return np.sum(np.abs(eigvals) < 1e-10)


def majorana_polarization(
    H: np.ndarray,
    threshold: float = 1e-7,
    axis: str = 'total',
    site: t.Optional[t.Union[int, str]] = 'avg'
):
    eigvals, eigvecs = np.linalg.eigh(H)
    zm = eigvecs[:, np.abs(eigvals) < threshold]
    if zm.shape[1] == 0:
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
    if site == 'all':
        return P_m
    else:
        raise ValueError('site must be one of "avg", "all", or an integer')


def majorana_polarization_site(zero_mode: np.ndarray, axis: str = 'total'):
    if axis == 'total':
        return np.mean(np.abs(zero_mode[1, :] * zero_mode[3, :] - zero_mode[0, :] * zero_mode[2, :]))
    if axis == 'x':
        return np.mean(np.real(zero_mode[1, :] * zero_mode[3, :] + zero_mode[0, :] * zero_mode[2, :]))
    if axis == 'y':
        return np.mean(np.imag(zero_mode[1, :] * zero_mode[3, :] - zero_mode[0, :] * zero_mode[2, :]))


def plot_eigvals(model: Hamiltonian, xaxis: str, xparams: t.List[t.Any], params: t.Dict[str, t.Any], filename: str, **kwargs: t.Dict[str, t.Any]):
    energies = []
    for x in xparams:
        model_params = params.copy()
        model_params[xaxis] = x
        ladder = model(**model_params)
        energies.append(np.linalg.eigvalsh(ladder.H))

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