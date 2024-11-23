from collections import defaultdict
from copy import deepcopy
from functools import reduce
import json
import os
import typing as t

import numpy as np
from scipy import sparse
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.transforms import Normalize
from tqdm import tqdm

from src.hamiltonian.hamiltonian import Hamiltonian, RepresentationMapping, REPRESENTATION_MAPPING_FUNCTION

DICTIONARY_NAME = 'dictionary.txt'
PARAMS_DICTIONARY_NAME = 'params_dictionary.txt'
MATRICES_DIR_NAME = 'matrices'
EIGVALS_DIR_NAME = 'eigvals'
EIGVEC_DIR_NAME = 'eigvec'


class HamiltionianDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        data_limit: t.Optional[int] = None,
        label_idx: t.Union[int, t.Tuple[int, int]] = 1,
        threshold: float = 1.e-5,
        eigvals: bool = False,
        eig_decomposition: bool = False,
        format: str = 'numpy',
        normalization_mean: t.Tuple[float, float] = (0., 0.),
        normalization_std: t.Tuple[float, float] = (1., 1.),
        representation_mapping: RepresentationMapping = RepresentationMapping.none,
        **kwargs,
    ):
        dic_path = os.path.join(data_dir, DICTIONARY_NAME)
        self.dictionary = self.load_dict(dic_path)
        self.data_dir = data_dir
        self.data_limit = data_limit
        self.label_idx = label_idx
        self.threshold = threshold
        self.eigvals = eigvals
        self.eig_dec = eig_decomposition
        self.eig_vals_num = kwargs.get('eigvals_num', 4)
        self.format = format
        self.gt_threshold = kwargs.get('gt_threshold', False)
        self.normalization = Normalize(normalization_mean, normalization_std)
        self.representation_mapping = representation_mapping
      
    def __len__(self) -> int:
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)


    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], t.Optional[t.Tuple[torch.Tensor, torch.Tensor]]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tensor = self.load_data(MATRICES_DIR_NAME, idx, self.format)
        if self.representation_mapping != RepresentationMapping.none:
            tensor = torch.from_numpy(REPRESENTATION_MAPPING_FUNCTION[self.representation_mapping](tensor.numpy()))
        tensor = torch.stack((tensor.real, tensor.imag), dim=0)
        tensor = self.normalization(tensor)

        if self.eig_dec:
            try:
                eigvals = self.load_data(EIGVALS_DIR_NAME, idx, 'numpy')
                eigvec = self.load_data(EIGVEC_DIR_NAME, idx, 'numpy')
                eig_dec = eigvals.real, eigvec
            except:
                complex_tensor = torch.complex(tensor[0], tensor[1])
                eigvals, eigvec = torch.linalg.eigh(complex_tensor)
                min_eigvals, min_eigvals_id = torch.topk(torch.abs(eigvals), self.eig_vals_num, largest=False)
                min_eigvec = eigvec[:, min_eigvals_id]
                min_eigvals = eigvals[min_eigvals_id] # because min_eigvals were absolute values
                eig_dec = min_eigvals.real, min_eigvec
                save_matrix(min_eigvals, self.data_dir, EIGVALS_DIR_NAME, self.dictionary[idx][0], format='numpy')
                save_matrix(min_eigvec, self.data_dir, EIGVEC_DIR_NAME, self.dictionary[idx][0], format='numpy')
        elif self.eigvals:
            try:
                eigvals = self.load_data(EIGVALS_DIR_NAME, idx, 'numpy')
                if len(eigvals) < self.eig_vals_num:
                    raise Exception()
                eig_dec = eigvals.real, torch.zeros((tensor.shape[0], tensor.shape[1]))
            except:
                complex_tensor = torch.complex(tensor[0], tensor[1])
                eigvals = torch.linalg.eigvalsh(complex_tensor)
                min_eigvals, min_eigvals_id = torch.topk(torch.abs(eigvals), self.eig_vals_num, largest=False)
                min_eigvals = eigvals[min_eigvals_id] # because min_eigvals were absolute values
                eig_dec = min_eigvals.real, torch.zeros((tensor.shape[0], tensor.shape[1]))
                save_matrix(min_eigvals, self.data_dir, EIGVALS_DIR_NAME, self.dictionary[idx][0], format='numpy')
        else:
            eig_dec = torch.zeros((1, tensor.shape[1])), torch.zeros((tensor.shape[0], tensor.shape[1]))

        label = self.get_label(idx, self.label_idx)
        label = torch.tensor(label)

        return (tensor, label), eig_dec
    
    def get_label(self, idx: int, label_idx: t.Union[int, t.Tuple, t.List]) -> t.Union[float, t.List[float]]:
        if type(label_idx) == int:
            label = [float(self.dictionary[idx][label_idx])]
        elif type(label_idx) == tuple:
            label = reduce(lambda x, y: x * y, [l for i in label_idx for l in self.get_label(idx, i)])
            # label = float(self.dictionary[idx][label_idx[0]]) * float(self.dictionary[idx][label_idx[1]])
            if self.gt_threshold:
                label = [1. if label > self.threshold else 0.]
            else:
                label = [1. if label < -self.threshold else 0.]
        elif type(label_idx) == list:
            label = [self.get_label(idx, i) for i in label_idx]
            # make list flat if it is nested
            flat_label = []
            for sublist in label:
                if type(sublist) == list:
                    flat_label.extend(sublist)
                else:
                    flat_label.append(sublist)
            label = flat_label
        else:
            raise ValueError("Wrong label_idx type")
        return label


    def load_dict(self, filepath: str) -> t.List[t.List[str]]:
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()
        parsed_data = [row.rstrip("\n").split(', ') for row in data]
        return parsed_data
    

    def load_data(self, dir: str, idx: int, format: str):
        if format == 'numpy':
            data_path = os.path.join(self.data_dir, dir, self.dictionary[idx][0] + '.npy')
            data = np.load(data_path)
        elif format == 'csr':
            data_path = os.path.join(self.data_dir, dir, self.dictionary[idx][0] + '.npz')
            data = sparse.load_npz(data_path)
            data = data.toarray()
        else:
            raise ValueError("Wrong format")
        return torch.from_numpy(data).type(torch.complex64)
    

class HamiltionianParamsDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        data_limit: t.Optional[int] = None,
        label_key: t.Union[str, t.List[str]] = 'increase_potential_at_edges',
        threshold: float = 1.e-5,
        format: str = 'numpy',
        **kwargs,
    ):
        dic_path = os.path.join(data_dir, PARAMS_DICTIONARY_NAME)
        self.dictionary = self.load_dict(dic_path)
        self.data_dir = data_dir
        self.data_limit = data_limit
        self.label_key = label_key
        self.threshold = threshold
        self.format = format

    def __len__(self) -> int:
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)

    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tensor = self.load_data(MATRICES_DIR_NAME, idx, self.format)
        tensor = torch.stack((tensor.real, tensor.imag), dim=0)
        label = self.parse_label(idx)
        return (tensor, label), torch.zeros((1, tensor.shape[1]))

    def load_dict(self, filepath: str) -> t.List[t.List[str]]:
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()
        parsed_data = [row.rstrip("\n").split(', ', maxsplit=1) for row in data]
        return parsed_data
    
    def load_data(self, dir: str, idx: int, format: str) -> torch.Tensor:
        if format == 'numpy':
            data_path = os.path.join(self.data_dir, dir, self.dictionary[idx][0] + '.npy')
            data = np.load(data_path)
        elif format == 'csr':
            data_path = os.path.join(self.data_dir, dir, self.dictionary[idx][0] + '.npz')
            data = sparse.load_npz(data_path)
            data = data.toarray()
        else:
            raise ValueError("Wrong format")
        return torch.from_numpy(data).type(torch.complex64)
    
    def parse_label(self, idx: int) -> torch.Tensor:
        if type(self.label_key) == str:
            label = json.loads(self.dictionary[idx][1])[self.label_key]
        elif type(self.label_key) == list:
            label = 1.
            for key in self.label_key:
                label *= abs(json.loads(self.dictionary[idx][1])[key])
        else:
            raise ValueError("Wrong label_key type")
        return torch.tensor(label)
    

class NoisyParametersHamiltonianDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        hamiltionian_class: t.Type[Hamiltonian],
        params_noise_config: t.Dict[str, float],
        data_limit: t.Optional[int] = None,
        label_idx: t.Union[int, t.Tuple[int, int]] = 1,
        threshold: float = 1.e-5,
        format: str = 'numpy',
        normalization_mean: t.Tuple[float, float] = (0., 0.),
        normalization_std: t.Tuple[float, float] = (1., 1.),
        representation_mapping: RepresentationMapping = RepresentationMapping.none,
        **kwargs,
    ):
        param_dict_path = os.path.join(data_dir, PARAMS_DICTIONARY_NAME)
        self.params_dictionary = self.load_params_dict(param_dict_path)
        label_dict_path = os.path.join(data_dir, DICTIONARY_NAME)
        self.dictionary = self.load_label_dict(label_dict_path)
        self.data_dir = data_dir
        self.data_limit = data_limit
        self.threshold = threshold
        self.format = format
        self.label_idx = label_idx
        self.hamiltionian_class = hamiltionian_class
        self.representation_mapping = representation_mapping
        self.normalization = Normalize(normalization_mean, normalization_std)
        self.params_noise_config = params_noise_config

    def load_label_dict(self, filepath: str) -> t.List[t.List[str]]:
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()
        parsed_data = [row.rstrip("\n").split(', ') for row in data]
        return parsed_data

    def load_params_dict(self, filepath: str) -> t.List[t.List[str]]:
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()
        parsed_data = [row.rstrip("\n").split(', ', maxsplit=1) for row in data]
        return parsed_data

    def __len__(self) -> int:
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)

    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Tuple[torch.Tensor, torch.Tensor]:
        hamiltonian_params = self.load_params(idx)
        tensor = self.generate_hamiltonian_tensor(hamiltonian_params)

        hamiltonian_noisy_params = self.add_noise_to_params(hamiltonian_params, self.params_noise_config)
        noisy_tensor = self.generate_hamiltonian_tensor(hamiltonian_noisy_params)

        label = self.get_label(idx, self.label_idx)
        return (tensor, noisy_tensor), label

    def generate_hamiltonian_tensor(self, hamiltonian_params):
        model = self.hamiltionian_class(**hamiltonian_params)
        tensor = model.get_hamiltonian_tensor(representation_mapping=self.representation_mapping)
        tensor = self.normalization(tensor)
        return tensor
    
    def load_params(self, idx: int) -> t.Dict[str, t.Any]:
        '''
        Example of dictionary entry (from QDH dataset):
        {
            "parameters": {
                "no_dots": 3,
                "no_levels": 1,
                "def_par": {
                    "mu_default": 0.0,
                    "mu_range": [-3.6749303600696764e-05, 3.6749303600696764e-05],
                    "dot_split": 3.6749303600696764e-05,
                    "t_default": 7.349860720139353e-06,
                    "t_range": [0.0, 3.6749303600696764e-05],
                    "b_default": 1.8374651800348382e-05,
                    "b_range": [-3.6749303600696764e-05, 3.6749303600696764e-05],
                    "d_default": 1.8374651800348382e-05,
                    "d_range": [0.0, 3.6749303600696764e-05],
                    "ph_d_default": 0.0,
                    "ph_d_range": [-3.141592653589793, 3.141592653589793],
                    "l_default": 0.6283185307179586,
                    "l_range": [0.0, 6.283185307179586],
                    "l_rho_default": 1.5707963267948966,
                    "l_rho_range": [0.0, 3.141592653589793],
                    "l_ksi_default": 0.0,
                    "l_ksi_range": [0.0, 6.283185307179586]
                },
                "mu": [4.2469372988207406e-06, 1.8172838637974256e-05, 2.5629978219913487e-05],
                "t": [7.349860720139353e-06, 7.10811336119953e-06, 2.3596521488823145e-06],
                "b": [4.667710259132149e-06, 4.667710259132149e-06, 4.667710259132149e-06],
                "d": [1.245189236429985e-05, 1.245189236429985e-05, 1.245189236429985e-05],
                "ph_d": 0.1446565075238544,
                "l": [0.6283185307179586, 0.6283185307179586, 0.6283185307179586],
                "l_rho": [1.5707963267948966, 1.5707963267948966, 1.5707963267948966],
                "l_ksi": [0.0, 0.0, 0.0]
            }
        }

        '''
        return json.loads(self.params_dictionary[idx][1])
    
    def add_noise_to_params(self, params: t.Dict[str, t.Any], params_noise_strength: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
        noisy_params = deepcopy(params)
        for key, value in params_noise_strength.items():
            if isinstance(value, dict):
                noisy_params[key] = self.add_noise_to_params(params[key], value)
            else:
                params_array = np.array(noisy_params[key])
                noisy_params[key] = (1 - value[0]) * params_array + value[0]*np.random.normal(0, value[1], params_array.shape)
        return noisy_params
    
    def get_label(self, idx: int, label_idx: t.Union[int, t.Tuple, t.List]) -> t.Union[float, t.List[float]]:
        if type(label_idx) == int:
            label = [float(self.dictionary[idx][label_idx])]
        elif type(label_idx) == tuple:
            label = reduce(lambda x, y: x * y, [l for i in label_idx for l in self.get_label(idx, i)])
            # label = float(self.dictionary[idx][label_idx[0]]) * float(self.dictionary[idx][label_idx[1]])
            if self.gt_threshold:
                label = [1. if label > self.threshold else 0.]
            else:
                label = [1. if label < -self.threshold else 0.]
        elif type(label_idx) == list:
            label = [self.get_label(idx, i) for i in label_idx]
            # make list flat if it is nested
            flat_label = []
            for sublist in label:
                if type(sublist) == list:
                    flat_label.extend(sublist)
                else:
                    flat_label.append(sublist)
            label = flat_label
        else:
            raise ValueError("Wrong label_idx type")
        return label

  
def generate_data(
    hamiltionian: t.Type[Hamiltonian],
    param_list: t.List[t.Dict[str, t.Any]],
    directory: str,
    eig_decomposition: bool = False,
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
