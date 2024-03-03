from copy import deepcopy
import os
import typing as t

import numpy as np
import matplotlib.pyplot as plt

from src.data_utils import Hamiltonian


def count_mzm_states(H: np.ndarray, threshold: float = 1.e-5):
    eigvals = np.linalg.eigvalsh(H)
    return np.sum(np.abs(eigvals) < threshold)


def calculate_gap(H: np.ndarray):
    eigvals = np.linalg.eigvalsh(H)
    negative_eigvals = eigvals[eigvals < 0]
    positive_eigvals = eigvals[eigvals > 0]
    return np.min(positive_eigvals) - np.max(negative_eigvals)


def calculate_mzm_main_bands_gap(H: np.ndarray, mzm_threshold: float = 1.e-5):
    eigvals = np.linalg.eigvalsh(H)
    mzms = eigvals[np.abs(eigvals) < mzm_threshold]
    not_mzms = eigvals[np.abs(eigvals) >= mzm_threshold]
    return np.min(np.abs(not_mzms)) - np.max(np.abs(mzms))


def are_majoranas_in_hamiltonian(H: np.ndarray, zm_threshold: float = 1.e-5, mzm_gap_threshold: float = 0.08):
    eigvals = np.linalg.eigvalsh(H)
    num_zm = np.sum(np.abs(eigvals) < zm_threshold)
    if num_zm == 0:
        return False

    mzms = eigvals[np.abs(eigvals) < zm_threshold]
    not_mzms = eigvals[np.abs(eigvals) >= zm_threshold]
    mzm_gap = np.min(np.abs(not_mzms)) - np.max(np.abs(mzms))
    if mzm_gap < mzm_gap_threshold:
        return False 
    return True    


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
        return 2*np.mean(np.abs(zero_mode[1, :] * zero_mode[2, :].conj() + zero_mode[0, :] * zero_mode[3, :].conj()))
    if axis == 'x':
        return 2*np.mean(np.real(zero_mode[1, :] * zero_mode[2, :].conj() + zero_mode[0, :] * zero_mode[3, :].conj()))
    if axis == 'y':
        return 2*np.mean(np.imag(zero_mode[1, :] * zero_mode[2, :].conj() + zero_mode[0, :] * zero_mode[3, :].conj()))


def plot_eigvals(model: Hamiltonian, xaxis: str, xparams: np.ndarray, filename: str, **kwargs: t.Dict[str, t.Any]):
    energies = []
    for x in xparams:
        ladder = deepcopy(model)
        ladder.set_parameter(xaxis, x)
        energies.append(np.linalg.eigvalsh(ladder.get_hamiltonian()))
    energies = np.array(energies)

    xnorm = None
    ynorm = None
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
    if 'ynorm' in kwargs:
        energies = energies / kwargs['ynorm']
        if kwargs['ynorm'] == np.pi:
            ynorm = 'π'
        else:
            ynorm = kwargs['ynorm']

    plt.plot(xparams, energies)
    if xnorm:
        plt.xlabel(f'{xaxis}/{xnorm}')
    else:
        plt.xlabel(f'{xaxis}')
    if ynorm:
        plt.ylabel(f'Energy/{ynorm}')
    else:
        plt.ylabel('Energy')
    plt.savefig(filename)
    plt.close()


def plot_eigvals_levels(
    model: Hamiltonian,
    save_path: str,
    **kwargs: t.Dict[str, t.Any],
):
    H = model.get_hamiltonian()
    eigvals = np.linalg.eigvalsh(H)
    
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
   
    xrange = [0, 10]
    for i in range(len(eigvals)):
        plt.plot(xrange, [eigvals[i], eigvals[i]])

    plt.xticks([])
    plt.ylabel('Energy')
    plt.savefig(save_path)
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

    string_num = kwargs.get('string_num', 1)

    for i in range(eigvecs.shape[1]):
        real = np.array([np.real(eigvecs[4*site + component, i]) for site in range(eigvecs.shape[0] // 4)])
        imag = np.array([np.imag(eigvecs[4*site + component, i]) for site in range(eigvecs.shape[0] // 4)])

        if string_num > 1:
            real = real.reshape((-1, string_num))
            imag = imag.reshape((-1, string_num))

            for j in range(string_num):
                site_plot(real[:, j], dirpath + f'/real_{i}_string_{j}.png', 'Eigenvalue: ' + str(eigvals[i]), 'Real part')
                site_plot(imag[:, j], dirpath + f'/imag_{i}_string_{j}.png', 'Eigenvalue: ' + str(eigvals[i]), 'Imaginary part')

        else:
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

    string_num = kwargs.get('string_num', 1)

    H = model.get_hamiltonian()
    eigvals, eigvecs = np.linalg.eigh(H)

    zm_eigvals = eigvals[np.abs(eigvals) < threshold]
    zm = eigvecs[:, np.abs(eigvals) < threshold]

    P_m_summed = np.zeros((H.shape[0] // (4 * string_num), string_num))
    for i in range(zm.shape[1]):
        zm_nambu = zm[:,i].reshape(-1,4)
        P_m = [
            majorana_polarization_site(
                np.expand_dims(zm_nambu[site], axis=1),
                axis=polaxis,
            ) 
            for site in range(H.shape[0] // 4)
        ]
        P_m = np.array(P_m).reshape((-1, string_num))
        P_m_summed += np.array(P_m)

        for j in range(string_num):
            site_plot(P_m[:, j], f'{dirpath}/polarization_{i}_string_{j}.png', f'Eigenvalue: {zm_eigvals[i]}', 'Majorana polarization', **kwargs)

    for j in range(string_num):
        site_plot(P_m_summed[:, j], f'{dirpath}/polarization_summed_string_{j}.png', 'Summed over eigenvalues', 'Majorana polarization', **kwargs)


def plot_site_matrix_elements(model: Hamiltonian, property_name: str, dirpath: str):
    # matrix structure
    # mu + B                   | S/2*(cos(phi)-isin(phi)) | 0                                | delta 
    # S/2*(cos(phi)+isin(phi)) | mu -B                    | -delta                           | 0
    # 0                        | -delta*                  | -mu -B*                          | -S/2*(cos(phi)+isin(phi))
    # delta*                   | 0                        |  -S/2*(cos(phi)-isin(phi))       |-mu +B*
    
    property_to_element_ids = {
        'potential': [(0, 0), (1, 1), (2, 2), (3, 3)],
        'magnetic_field': [(0, 0), (1, 1), (2, 2), (3, 3)],
        'spin': [(0, 1), (1, 0), (2, 3), (3, 2)],
        'delta': [(0, 3), (1, 2), (2, 1), (3, 0)],
        'interaction_i_j': [(0, 0), (1, 1), (2, 2), (3, 3)],
        'interaction_j_i': [(0, 0), (1, 1), (2, 2), (3, 3)],
    }
    property_to_sign = {
        'potential': [1, 1, -1, -1],
        'magnetic_field': [1, -1, -1, 1],
        'spin': [1, 1, -1, -1],
        'delta': [1, -1, -1, 1],
        'interaction_i_j': [1, 1, -1, -1],
        'interaction_j_i': [1, 1, -1, -1],
    }
    if property_name == 'interaction_i_j':
        site_shift = (0, 1)
    elif property_name == 'interaction_j_i':
        site_shift = (1, 0)
    else:
        site_shift = (0, 0)

    site_elements = []
    for element_ids, sign in zip(property_to_element_ids[property_name], property_to_sign[property_name]):
        matrix_elements = extract_matrix_elements(model, element_ids, site_shift)
        matrix_array = np.array(matrix_elements) * sign
        site_elements.append(np.abs(matrix_array))
    site_elements = np.stack(site_elements, axis=1)
    site_elements_mean = np.mean(site_elements, axis=1)
    site_elements_std = np.std(site_elements, axis=1)
    site_plot(site_elements_mean, f'{dirpath}/{property_name}.png', f'Averaged {property_name}', f'{property_name}', errorbar=site_elements_std)


def extract_matrix_elements(model: Hamiltonian, element_ids: t.Tuple[int, int], site_shift: t.Tuple[int, int] = (0, 0)):
    # element ids are (row, column) of the 4 x 4 matrix
    H = model.get_hamiltonian()
    sites_num = H.shape[0] // 4
    matrix_elements = [
        H[(4*(site_id + site_shift[0]) + element_ids[0]) % H.shape[0], (4*(site_id + site_shift[1]) + element_ids[1]) % H.shape[1]]
        for site_id in range(sites_num)
    ]
    return matrix_elements


def site_plot(values: np.ndarray, filename: str, title: str, ylabel: str, **kwargs: t.Dict[str, t.Any]):
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])

    if 'errorbar' in kwargs:
        plt.errorbar(range(len(values)), values, yerr=kwargs['errorbar'], ecolor='red')
    else:
        plt.plot(values)
    plt.title(title)
    plt.xlabel('Site')
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()


