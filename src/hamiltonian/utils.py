from copy import deepcopy
import os
import typing as t

import numpy as np
import matplotlib.pyplot as plt

from src.hamiltonian.hamiltonian import IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR, REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR, Hamiltonian, RepresentationMapping
from src.hamiltonian.hamiltonian_torch_handlers import get_strip, BlockExtractor
from src.hamiltonian.units import AtomicUnits


def count_mzm_states(H: np.ndarray, threshold: float = 1.e-5):
    eigvals = np.linalg.eigvalsh(H)
    return np.sum(np.abs(eigvals) < threshold)


def calculate_gap(H: np.ndarray):
    eigvals = np.linalg.eigvalsh(H)
    negative_eigvals = eigvals[eigvals < 0]
    positive_eigvals = eigvals[eigvals > 0]
    return np.min(positive_eigvals) - np.max(negative_eigvals)


def calculate_mzm_main_bands_gap(H: np.ndarray, mzm_threshold: t.Optional[float] = 1.e-5, num_majoranas: t.Optional[int] = None):
    eigvals = np.linalg.eigvalsh(H)
    if mzm_threshold is not None and num_majoranas is not None:
        mzms = np.sort(eigvals[np.abs(eigvals) < mzm_threshold])[:num_majoranas]
        not_mzms = eigvals[np.abs(eigvals) >= mzm_threshold]
    elif mzm_threshold is not None:
        mzms = eigvals[np.abs(eigvals) < mzm_threshold]
        not_mzms = eigvals[np.abs(eigvals) >= mzm_threshold]
    elif num_majoranas is not None:
        mzms = np.sort(np.abs(eigvals))[:num_majoranas]
        not_mzms = np.sort(np.abs(eigvals))[num_majoranas:]
    else:
        raise ValueError('Either mzm_threshold or num_majoranas must be provided')
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


def majoranization(H: np.ndarray, n_dots: int):
    # m1 = np.concatenate([np.array([1,1,-1,-1]), np.array([0,0,0,0]*(n_dots-1))]).T/2.
    # m2 = np.concatenate([np.array([0,0,0,0]*(n_dots-1)), np.array([1,1,-1,-1])]).T/2.
    m1 = np.concatenate([np.array([1,1,-1,-1]), np.array([0,0,0,0]*(n_dots-2)), np.array([1,1,1,1])]).T/2.
    m2 = np.concatenate([np.array([1,1,-1,-1])*-1., np.array([0,0,0,0]*(n_dots-2)), np.array([1,1,1,1])]).T/2.

    eigvals, eigvecs = np.linalg.eigh(H)
    ms = []
    ehs = []
    for ie, eig in enumerate(eigvals):
        ml = np.conj(eigvecs[:,ie])@m1
        mr = np.conj(eigvecs[:,ie])@m2
        zm = np.exp(-np.abs(eig)/(0.1/AtomicUnits.Eh))
        # ms.append(np.abs(ml+mr)*zm)  # mode projection on left + right majoranas
        ms.append(np.abs(ml-mr)*zm)  # mode projection on left + right majoranas

        eh = (np.abs(eigvecs[:,ie])**2).reshape(-1,2,2).sum(axis=(0,2))
        ehs.append(np.amax([0., eh[0]*eh[1]*4-0.5])*2.)  # smaller than 0.5 are filtered out

    ms = np.array(ms)[np.abs(eigvals).argsort()]  # # sort MZM_i using |E_i|
    ehs = np.array(ehs)[np.abs(eigvals).argsort()]  # same here

    # theoretical max
    """
    max_value = 2
    Reasoning:
    - max of ms[0] + ms[1] is 2 (both modes fully localized on the edges)
    - min of sum of other ms is 0 (no other modes)
    - max of ehs[0] is 1 (mode fully electron or holeic)
    Thus max majoranization is 2*1 / 2 = 1
    """
    # max_value = 2
    max_value = 2*np.sqrt(2)

    majoranization = np.amax([0., ms[0]+ms[1]-ms[2:].sum()])*ehs[0] / max_value
    return majoranization


def plot_eigvals(model: Hamiltonian, xaxis: str, xparams: np.ndarray, filename: str, color: t.Optional[str] = None, majoranization: bool = False, **kwargs: t.Dict[str, t.Any]):
    block_dim = 4
    representation_mapping = kwargs.get('representation_mapping', RepresentationMapping.default)
    energies = []
    color_values = []
    majoranization_values = []
    m1 = np.concatenate([np.array([1,1,-1,-1]), np.array([0,0,0,0]*(model.num_sites-2)), np.array([1,1,1,1])]).T/2.
    m2 = np.concatenate([np.array([1,1,-1,-1])*-1., np.array([0,0,0,0]*(model.num_sites-2)), np.array([1,1,1,1])]).T/2.
    # m1 = np.concatenate([np.array([1,1,-1,-1]), np.array([0,0,0,0]*(model.num_sites-1))]).T/2.
    # m2 = np.concatenate([np.array([0,0,0,0]*(model.num_sites-1)), np.array([1,1,-1,-1])]).T/2.

    m_color = 'darkred'

    for x in xparams:
        current_model = deepcopy(model)
        current_x_value = current_model.get_parameter(xaxis)
        current_model.set_parameter(xaxis, current_x_value + x)
        H = current_model.get_hamiltonian(representation_mapping)
        eigs, eigvecs = np.linalg.eigh(H)
        energies.append(eigs)
        colors_eigs = np.zeros_like(eigs)

        ms = []
        ehs = []
        for i in range(len(eigs)):
            eigvec_module = eigvecs[:,i].real ** 2 + eigvecs[:,i].imag ** 2
            eigvec_module_sites = eigvec_module.reshape(-1, block_dim)
            
            if color == 'occupations':
                occs = np.sum(eigvec_module_sites, axis=1)
                color_val = (occs[0] + occs[-1])
                color_label = 'Occupations'
                vmin = 0.
                vmax = 1.
            
            elif color == 'electron-hole-diff':
                eh_diff = eigvec_module_sites[:, 0] + eigvec_module_sites[:, 1] - eigvec_module_sites[:, 2] - eigvec_module_sites[:, 3]
                color_val = eh_diff.sum()
                color_label = 'Electron-hole difference'
                vmin = -1.
                vmax = 1.

            else:
                color_val = 1
                vmin = None
                vmax = None

            if majoranization:
                ml = np.conj(eigvecs[:,i])@m1
                mr = np.conj(eigvecs[:,i])@m2
                zm = np.exp(-np.abs(eigs[i])/(0.1/AtomicUnits.Eh))
                ms.append(np.abs(ml-mr)*zm)  # mode projection on left + right majoranas

                eh = (np.abs(eigvecs[:,i])**2).reshape(-1,2,2).sum(axis=(0,2))
                ehs.append(np.amax([0., eh[0]*eh[1]*4-0.5])*2.)  # smaller than 0.5 are filtered out

            colors_eigs[i] = color_val
        color_values.append(colors_eigs)

        if majoranization:
            ms = np.array(ms)[np.abs(eigs).argsort()]  # # sort MZM_i using |E_i|
            ehs = np.array(ehs)[np.abs(eigs).argsort()]  # same here

            max_value = 2*np.sqrt(2)
            majoranization_val = np.amax([0., ms[0]+ms[1]-ms[2:].sum()])*ehs[0] / max_value
            majoranization_values.append(majoranization_val)

    energies = np.array(energies)
    color_values = np.array(color_values)

    current_x_value = model.get_parameter(xaxis)
    xparams = xparams + np.mean(current_x_value)

    ax = kwargs.get('ax', None)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    xnorm = None
    ynorm = None
    if 'ylim' in kwargs:
        ax.set_ylim(kwargs['ylim'])
    if 'xlim' in kwargs:
        ax.set_xlim(kwargs['xlim'])
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

    if 'title' in kwargs:
        ax.set_title(kwargs['title'])

    # set min and max for color scale
    if color == "occupations":
        cmap = "viridis"
    elif color == "electron-hole-diff":
        cmap = "bwr_r"
    else:
        cmap = None
    plot = ax.scatter(np.expand_dims(xparams, axis=1).repeat(energies.shape[1], axis=1), energies, c=color_values, vmin=vmin, vmax=vmax, cmap=cmap, s=.2)    
    if kwargs.get('left_label', True):
        ax.set_ylabel('$E$ (meV)')
    ax.yaxis.set_label_coords(-0.16, 0.5)
    ax.set_yticks([-.5, 0., .5])

    ax.set_xlim(xparams[0], xparams[-1])
    x_vmin, x_vmax = ax.get_xlim()
    x_mid = 0.5 * (x_vmin + x_vmax)
    ax.set_xticks([ x_vmin, x_mid, x_vmax ])
    ax.set_xticklabels([ f'{x_vmin:.2f}', f'{x_mid:.2f}', f'{x_vmax:.2f}'])
    
    x_vline = kwargs.get('x_vline', x_mid)
    ax.axvline(x_vline, c='grey', ls='--')

    if kwargs.get('right_label', True):
        ax.xaxis.get_majorticklabels()[0].set_horizontalalignment('left')
    else:
        ax.xaxis.get_majorticklabels()[2].set_horizontalalignment('right')

    ax.tick_params(axis='x', pad=10)

    # Add second axis for majoranization
    if majoranization:
        ax2 = ax.twinx()
        if kwargs.get('right_label', True):
            ax2.set_ylabel(r'$\mathcal{M}$', color=m_color, rotation=0)
            ax2.yaxis.set_label_coords(1.2, 0.53)  # Move ylabel further right
            ax2.tick_params(axis='y', labelcolor=m_color)
        else:
            ax2.set_axis_off()

        # Set the color of the second axis to red and plot majoranization values
        ax2.plot(xparams, np.array(majoranization_values), color=m_color)
        ax2.set_ylim(0, 1.)  # Set y-axis limits for majoranization


    # if xnorm:
    #     plt.xlabel(f'{xaxis}/{xnorm}')
    # else:
    if xaxis == 'potential' or xaxis == 'mu':
        ax.set_xlabel('$\mu$ (meV)')
    else:
        ax.set_xlabel(f'{xaxis}')

    ax.xaxis.set_label_coords(0.5, -0.16)


    # ax.text(-0.15, 0.485, "E", color='black', rotation='vertical', transform=ax.transAxes)

    # # Plot legend for majoranization
    # if majoranization:
    #     ax.text(-0.15, 0.515, "(M)", color='red', rotation='vertical', transform=ax.transAxes)

    if fig is not None:
        plt.colorbar(plot, label = color_label, orientation='horizontal', pad=0.5, ax=ax)
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    return plot


def plot_eigvals_levels(
    model: Hamiltonian,
    save_path: str,
    **kwargs: t.Dict[str, t.Any],
):
    representation_mapping = kwargs.get('representation_mapping', RepresentationMapping.default)
    H = model.get_hamiltonian(representation_mapping)
    eigvals = np.linalg.eigvalsh(H)
    
    if 'title' in kwargs:
        plt.title(kwargs['title'])

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

    if 'title' in kwargs:
        plt.title(kwargs['title'])

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
