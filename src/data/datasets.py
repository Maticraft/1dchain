from copy import deepcopy
from src.data.utils import DICTIONARY_NAME, EIGVALS_DIR_NAME, EIGVEC_DIR_NAME, CONDUCTANCE_CMAP_0_DIR_NAME, CONDUCTANCE_CMAP_1_DIR_NAME, MATRICES_DIR_NAME, PARAMS_DICTIONARY_NAME, save_matrix
from src.hamiltonian.conductance import generate_conductance_tensor
from src.hamiltonian.hamiltonian import REPRESENTATION_MAPPING_FUNCTION, Hamiltonian, RepresentationMapping
from src.hamiltonian.quantum_dots_chain import MZM_THRESHOLD

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import json
import os
import typing as t
from functools import reduce

from src.hamiltonian.utils import calculate_gap, majoranization


class HamiltionianDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        data_limit: t.Optional[int] = None,
        label_idx: t.Union[int, t.Tuple[int, int]] = 1,
        threshold: float = 1.e-5,
        eigvals: bool = False,
        eig_decomposition: bool = False,
        cmap: str = 'none',
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
        self.cmap = cmap

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

        if self.cmap != 'none':
            if self.cmap == 'cmap0':
                cmap = self.load_data(CONDUCTANCE_CMAP_0_DIR_NAME, idx, self.format)
            elif self.cmap == 'cmap1':
                cmap = self.load_data(CONDUCTANCE_CMAP_1_DIR_NAME, idx, self.format)
            else:
                raise ValueError(f"Wrong cmap value: {self.cmap}")
            label = cmap.to(torch.float32).unsqueeze(0)
        else:
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


class HamiltonianFromParametersDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        hamiltionian_class: t.Type[Hamiltonian],
        params_noise_config: t.Optional[t.Dict[str, float]] = None,
        data_limit: t.Optional[int] = None,
        label_idx: t.Union[int, t.Tuple[int, int]] = 1,
        threshold: float = 1.e-5,
        format: str = 'numpy',
        normalization_params: t.List[t.Tuple[t.Tuple[float, ...], t.Tuple[float, ...]]] = None,
        representation_mapping: RepresentationMapping = RepresentationMapping.none,
        conductance_config: t.Optional[t.Dict[str, t.Any]] = None,
        random_noise: bool = False,
        assert_noise_majoranas_destruction: bool = False,
        n_dots: int = 3,
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
        self.normalization = None
        if normalization_params:
            self.normalization = [Normalize(mean, std) for mean, std in normalization_params]
        self.params_noise_config = params_noise_config
        self.conductance_config = conductance_config
        self.random_noise = random_noise
        self.assert_noise_majoranas_destruction = assert_noise_majoranas_destruction
        self.n_dots = n_dots

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
        h_tensor = self.generate_hamiltonian_tensor(hamiltonian_params)
        tensors = [h_tensor]
        if self.conductance_config is not None:
            cmap = self._generate_input_conductance(hamiltonian_params, self.conductance_config)
            tensors.append(cmap)
        else:
            tensors.append(h_tensor)

        if self.random_noise:
            noise_amplitude = abs(np.random.normal(loc=0., scale=0.1))
            params_noise_config = self.adjust_params_noise_strength(self.params_noise_config, noise_amplitude)
        else:
            noise_amplitude = 0.
            params_noise_config = self.params_noise_config

        if self.params_noise_config is not None:
            hamiltonian_params = self.add_noise_to_params(hamiltonian_params, params_noise_config)
            
            try:
                if self.assert_noise_majoranas_destruction:
                    model = self.hamiltionian_class(**hamiltonian_params)
                    tensor = model.get_hamiltonian_tensor()
                    tensor_complex = torch.complex(tensor[0], tensor[1])                
                    band_gap = calculate_gap(tensor_complex.numpy())
                    label = majoranization(tensor_complex.numpy(), self.n_dots)
                    are_majoranas_present = (label > 0.0) or (band_gap < 2*MZM_THRESHOLD)
                    
                    it = 0
                    while (are_majoranas_present and (it < 100)):
                        hamiltonian_params = self.add_noise_to_params(hamiltonian_params, params_noise_config)
                        model = self.hamiltionian_class(**hamiltonian_params)
                        tensor = model.get_hamiltonian_tensor()
                        tensor_complex = torch.complex(tensor[0], tensor[1])                
                        band_gap = calculate_gap(tensor_complex.numpy())
                        label = majoranization(tensor_complex.numpy(), self.n_dots)
                        are_majoranas_present = (label > 0.0) or (band_gap < 2*MZM_THRESHOLD)
                        it += 1
            except:
                pass

            noisy_tensor = self.generate_hamiltonian_tensor(hamiltonian_params)
            tensors.append(noisy_tensor)
        else:
            tensors.append(h_tensor)

        if self.conductance_config is not None:
            cmap = self._generate_input_conductance(hamiltonian_params, self.conductance_config)
            tensors.append(cmap)
        else:
            tensors.append(tensors[2])

        label = self.get_label(idx, self.label_idx), noise_amplitude
        if self.normalization:
            tensors = [normalize(tensor) for tensor, normalize in zip(tensors, self.normalization)]

        # tensors [normal, target_conductance, noisy, noisy_conductance]
        return tensors, label

    def _generate_input_conductance(
        self,
        hamiltonian_params: t.Dict[str, t.Any],
        conductance_config: t.Dict[str, t.Any],
    ):
        model = self.hamiltionian_class(**hamiltonian_params)
        tensor = model.get_hamiltonian_tensor()
        tensor_complex = torch.complex(tensor[0], tensor[1])
        cmap = generate_conductance_tensor(tensor_complex, conductance_config)
        return cmap

    def generate_hamiltonian_tensor(self, hamiltonian_params):
        model = self.hamiltionian_class(**hamiltonian_params)
        tensor = model.get_hamiltonian_tensor(representation_mapping=self.representation_mapping)
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
    
    def adjust_params_noise_strength(self, params_noise_strength: t.Dict[str, t.Any], noise_amplitude: float) -> t.Dict[str, t.Any]:
        new_noise_strength = {}
        for key, value in params_noise_strength.items():
            if isinstance(value, dict):
                new_noise_strength[key] = self.adjust_params_noise_strength(value, noise_amplitude)
            else:
                new_noise_strength[key] = (noise_amplitude * value[0], value[1])
        return new_noise_strength

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