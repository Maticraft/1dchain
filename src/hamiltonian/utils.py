from copy import deepcopy
import os
import typing as t

import numpy as np
import matplotlib.pyplot as plt

from src.hamiltonian.hamiltonian import IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR, REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR, Hamiltonian, RepresentationMapping
from src.hamiltonian.hamiltonian_torch_handlers import get_strip, BlockExtractor


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
    site: t.Optional[t.Union[int, str]] = 'avg',
    representation_mapping: RepresentationMapping = RepresentationMapping.default
):
    eigvals, eigvecs = np.linalg.eigh(H)
    zm = eigvecs[:, np.abs(eigvals) < threshold]

    if zm.shape[1] == 0:
        if site == 'all':
            return {'all': 0}
        return 0.

    if type(site) == int:
        zm_site = zm[4*site:4*(site+1), :]
        return majorana_polarization_site(zm_site, axis=axis, representation_mapping=representation_mapping)

    P_m = {}
    for i in range(zm.shape[0] // 4):
        zm_site_i = zm[4*i:4*(i+1), :]
        P_m[i] = majorana_polarization_site(zm_site_i, axis=axis, representation_mapping=representation_mapping)
        
    if site == 'avg':
        return np.mean(list(P_m.values()))
    if site == 'sum':
        return np.sum(list(P_m.values()))
    elif site == 'all':
        return P_m
    else:
        raise ValueError('site must be one of "avg", "all", or an integer')


def majorana_polarization_site(zero_mode: np.ndarray, axis: str = 'total', representation_mapping: RepresentationMapping = RepresentationMapping.default):
    if axis == 'total':
        return 2*np.mean(np.abs(majorana_polarization_product(zero_mode, representation_mapping)))
    if axis == 'x':
        return 2*np.mean(np.real(majorana_polarization_product(zero_mode, representation_mapping)))
    if axis == 'y':
        return 2*np.mean(np.imag(majorana_polarization_product(zero_mode, representation_mapping)))
    

def majorana_polarization_product(zero_mode: np.ndarray, representation_mapping: RepresentationMapping = RepresentationMapping.default):
    if representation_mapping == RepresentationMapping.majorana_plus_minus_up_down:
        return zero_mode[1, :] * zero_mode[3, :].conj() - zero_mode[0, :] * zero_mode[2, :].conj() # just guessing
    if representation_mapping == RepresentationMapping.second_quantized_polarization:
        '''Seems invalid for QDH (that's good actually)'''
        return zero_mode[1, :] * zero_mode[2, :].conj() + zero_mode[0, :] * zero_mode[3, :].conj()
    if representation_mapping == RepresentationMapping.second_quantized_plus_minus_up_down:
        return zero_mode[1, :] * zero_mode[3, :].conj() - zero_mode[0, :] * zero_mode[2, :].conj() # reversing conjugate makes the polarization flip signs - to discuss with Jarek
    else:
        raise ValueError(f'Representation mapping {representation_mapping} not supported')


def plot_eigvals(model: Hamiltonian, xaxis: str, xparams: np.ndarray, filename: str, **kwargs: t.Dict[str, t.Any]):
    representation_mapping = kwargs.get('representation_mapping', RepresentationMapping.default)
    energies = []
    for x in xparams:
        ladder = deepcopy(model)
        ladder.set_parameter(xaxis, x)
        H = ladder.get_hamiltonian(representation_mapping)
        energies.append(np.linalg.eigvalsh(H))
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
    representation_mapping = kwargs.get('representation_mapping', RepresentationMapping.default)
    H = model.get_hamiltonian(representation_mapping)
    eigvals = np.linalg.eigvalsh(H)
    
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])

    ynorm = None
    if 'ynorm' in kwargs:
        eigvals = eigvals / kwargs['ynorm']
        if kwargs['ynorm'] == np.pi:
            ynorm = 'π'
        else:
            ynorm = kwargs['ynorm']
   
    xrange = [0, 10]
    for i in range(len(eigvals)):
        plt.plot(xrange, [eigvals[i], eigvals[i]])

    plt.xticks([])
    if ynorm:
        plt.ylabel(f'Energy/{ynorm}')
    else:
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

    representation_mapping = kwargs.get('representation_mapping', RepresentationMapping.default)
    H = model.get_hamiltonian(representation_mapping)
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
                representation_mapping=representation_mapping
            ) 
            for site in range(H.shape[0] // 4)
        ]
        P_m = np.array(P_m).reshape((-1, string_num))
        P_m_summed += np.array(P_m)

        for j in range(string_num):
            site_plot(P_m[:, j], f'{dirpath}/polarization_{i}_string_{j}.png', f'Eigenvalue: {zm_eigvals[i]}', 'Majorana polarization', **kwargs)

    for j in range(string_num):
        site_plot(P_m_summed[:, j], f'{dirpath}/polarization_summed_string_{j}.png', 'Summed over eigenvalues', 'Majorana polarization', **kwargs)


def plot_site_varying_matrix_elements(model: Hamiltonian, property_name: str, dirpath: str, representation_mapping: RepresentationMapping = RepresentationMapping.default):
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
        hamiltonian_matrix = model.get_hamiltonian(representation_mapping=representation_mapping)
        matrix_elements = extract_matrix_elements(hamiltonian_matrix, element_ids, site_shift)
        matrix_array = np.array(matrix_elements) * sign
        site_elements.append(np.abs(matrix_array))
    site_elements = np.stack(site_elements, axis=1)
    site_elements_mean = np.mean(site_elements, axis=1)
    site_elements_std = np.std(site_elements, axis=1)
    site_plot(site_elements_mean, f'{dirpath}/{property_name}.png', f'Averaged {property_name}', f'{property_name}', errorbar=site_elements_std)


def extract_matrix_elements(hamiltonian_matrix: np.ndarray, element_ids: t.Tuple[int, int], site_shift: t.Tuple[int, int] = (0, 0), block_size: int = 4):
    # element ids are (row, column) of the 4 x 4 matrix
    sites_num = hamiltonian_matrix.shape[0] // block_size
    matrix_elements = [
        hamiltonian_matrix[(block_size*(site_id + site_shift[0]) + element_ids[0]) % hamiltonian_matrix.shape[0], (block_size*(site_id + site_shift[1]) + element_ids[1]) % hamiltonian_matrix.shape[1]]
        for site_id in range(sites_num)
    ]
    return matrix_elements


def plot_site_constant_matrix_elements(model: Hamiltonian, property_name: str, dirpath: str, **kwargs: t.Dict[str, t.Any]):
    representation_mapping = kwargs.get('representation_mapping', RepresentationMapping.default)
    try:
        site_real_elements = extract_property_strip(model, property_name, part='real', representation_mapping=representation_mapping)
        site_plot(site_real_elements, f'{dirpath}/{property_name}_real.png', f'{property_name} real part', f'{property_name} real part', **kwargs)
    except KeyError:
        pass

    try:
        site_imag_elements = extract_property_strip(model, property_name, part='imag', representation_mapping=representation_mapping)
        site_plot(site_imag_elements, f'{dirpath}/{property_name}_imag.png', f'{property_name} imaginary part', f'{property_name} imaginary part', **kwargs)
    except KeyError:
        pass


def plot_interaction_constant_matrix_elements(model: Hamiltonian, property_name: str, dirpath: str, **kwargs: t.Dict[str, t.Any]):
    interaction_level = kwargs.get('interaction_level', 1)
    representation_mapping = kwargs.get('representation_mapping', RepresentationMapping.default)
    try:
        site_real_elements = extract_property_strip(model, property_name, part='real', interaction_level=interaction_level, representation_mapping=representation_mapping)
        site_plot(site_real_elements, f'{dirpath}/{property_name}_real.png', f'{property_name} real part', f'{property_name} real part', **kwargs)
    except KeyError:
        pass

    try:
        site_imag_elements = extract_property_strip(model, property_name, part='imag', interaction_level=interaction_level, representation_mapping=representation_mapping)
        site_plot(site_imag_elements, f'{dirpath}/{property_name}_imag.png', f'{property_name} imaginary part', f'{property_name} imaginary part', **kwargs)
    except KeyError:
        pass


def extract_property_strip(model: Hamiltonian, property_name: str, part: str = 'real', interaction_level: int = 0, block_size: int = 4, representation_mapping: RepresentationMapping = RepresentationMapping.default):
    torch_hamiltonian = model.get_hamiltonian_tensor(representation_mapping).unsqueeze(0)
    strip = get_strip(torch_hamiltonian, interaction_level, fill_mode='hamiltonian', block_size=block_size)
    if part == 'real':
        property_block_name = REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR[property_name]
        property_strip = BlockExtractor.extract_block_sequences(strip[:, 0], [property_block_name])
    elif part == 'imag':
        property_block_name = IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR[property_name]
        property_strip = BlockExtractor.extract_block_sequences(strip[:, 1], [property_block_name])
    else:
        raise ValueError(f'Part: {part} not implemented')
    return property_strip.squeeze().numpy()


def site_plot(values: np.ndarray, filename: str, title: str, ylabel: str, **kwargs: t.Dict[str, t.Any]):
    if 'ylim' in kwargs:
        plt.ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        plt.xlim(kwargs['xlim'])

    ynorm = None
    if 'ynorm' in kwargs:
        values = values / kwargs['ynorm']
        if 'errorbar' in kwargs:
            kwargs['errorbar'] = kwargs['errorbar'] / kwargs['ynorm']

    if 'errorbar' in kwargs:
        plt.errorbar(range(len(values)), values, yerr=kwargs['errorbar'], ecolor='red')
    else:
        plt.plot(values)
    plt.title(title)
    plt.xlabel('Site')

    if ynorm:
        plt.ylabel(f'{ylabel}/{ynorm}')
    else:
        plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()
