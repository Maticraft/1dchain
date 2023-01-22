import typing as t

import numpy as np
from data_utils import Hamiltonian
import matplotlib.pyplot as plt


def count_mzm_states(H: np.ndarray):
    eigvals = np.linalg.eigvalsh(H)
    return np.sum(np.abs(eigvals) < 1e-10)


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