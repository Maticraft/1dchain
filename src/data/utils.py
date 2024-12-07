from collections import defaultdict
import json
import os
import typing as t

import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Normalize
from tqdm import tqdm

from src.hamiltonian.hamiltonian import Hamiltonian, RepresentationMapping
from src.hamiltonian.conductance import Transport

DICTIONARY_NAME = 'dictionary.txt'
PARAMS_DICTIONARY_NAME = 'params_dictionary.txt'
MATRICES_DIR_NAME = 'matrices'
EIGVALS_DIR_NAME = 'eigvals'
EIGVEC_DIR_NAME = 'eigvec'
CONDUCTANCE_CMAP_0_DIR_NAME = 'conductance_cmap_0'
CONDUCTANCE_CMAP_1_DIR_NAME = 'conductance_cmap_1'


def generate_data(
    hamiltionian: t.Type[Hamiltonian],
    param_list: t.List[t.Dict[str, t.Any]],
    directory: str,
    eig_decomposition: bool = False,
    conductance_config: t.Optional[t.Dict[str, t.Any]] = None,
    format: str = 'numpy',
    representation: RepresentationMapping = RepresentationMapping.default,
):
    for i, params in tqdm(enumerate(param_list), 'Generating data'):
        idx = i
        filename = 'data_' + str(idx)
        model = hamiltionian(**params)
        matrix = model.get_hamiltonian(representation)
        try:
            label = model.get_label()
        except:
            continue

        if eig_decomposition:
            try:
                eigvals, eigvec = np.linalg.eigh(matrix)
            except:
                continue
        else:
            eigvals, eigvec = None, None

        if conductance_config is not None:
            transport = Transport(model, conductance_config['gamma'])
            if 'cmap0' in conductance_config:
                cmap0 = transport.c_map0(**conductance_config['cmap0'])
                save_matrix(cmap0, directory, CONDUCTANCE_CMAP_0_DIR_NAME, filename, format)
            if 'cmap1' in conductance_config:
                cmap1 = transport.c_map1(**conductance_config['cmap1'])
                save_matrix(cmap1, directory, CONDUCTANCE_CMAP_1_DIR_NAME, filename, format)

        save_data(matrix, label, directory, filename, eigvals, eigvec, format, params)


def save_data(
    matrix: np.ndarray,
    label: str,
    root_dir: str,
    filename: str,
    eigvals: t.Optional[np.ndarray] = None,
    eigvec: t.Optional[np.ndarray] = None,
    format: str = 'numpy',
    params: t.Optional[t.Dict[str, t.Any]] = None,
):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)

    save_matrix(matrix, root_dir, MATRICES_DIR_NAME, filename, format)

    if eigvals is not None:
        save_matrix(eigvals, root_dir, EIGVALS_DIR_NAME, filename, format)
    if eigvec is not None:
        save_matrix(eigvec, root_dir, EIGVEC_DIR_NAME, filename, format)

    with open(os.path.join(root_dir, DICTIONARY_NAME), 'a') as dictionary:
        dictionary.write(f'{filename}, {label}\n')

    if params is not None:
        with open(os.path.join(root_dir, PARAMS_DICTIONARY_NAME), 'a') as params_file:
            params_str = json.dumps(params)
            params_file.write(f'{filename}, {params_str}\n')


def save_matrix(matrix: np.ndarray, root_dir: str, folder_name: str, file_name: str, format: str = 'numpy'):
    matrix_dir = os.path.join(root_dir, folder_name)
    if not os.path.isdir(matrix_dir):
        os.makedirs(matrix_dir)
    if format == 'numpy':
        matrix_name = os.path.join(matrix_dir, file_name + '.npy')
        np.save(matrix_name, matrix)
    elif format == 'csr':
        matrix_name = os.path.join(matrix_dir, file_name + '.npz')
        sparse.save_npz(matrix_name, sparse.csr_matrix(matrix))
    else:
        raise ValueError("Wrong format")
    

def calculate_mean_and_std(
    data_loader: DataLoader,
    device: torch.device,
    callable: t.Optional[t.Callable] = None
):
    # calculate latent space distribution (mean and std)
    mean = defaultdict(float)
    std = defaultdict(float)
    for (data, _), _ in tqdm(data_loader, 'Collecting data statistics...'):
        data = data.to(device)
        if callable is not None:
            data = callable(data)
        for channel in range(data.shape[1]):
            mean[channel] += data[:, channel].mean().item()
            std[channel] += data[:, channel].std().item()
    for channel in mean.keys():
        mean[channel] /= len(data_loader)
        std[channel] /= len(data_loader)
    return tuple(mean.values()), tuple(std.values())


class Denormalize(Normalize):
    def __init__(self, mean: t.Tuple[float, ...], std: t.Tuple[float, ...]):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        super().__init__((-mean/std).tolist(), (1./std).tolist())
