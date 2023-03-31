import abc
import os
import typing as t

import numpy as np
from scipy import sparse
from torch.utils.data import Dataset
import torch
from tqdm import tqdm

DICTIONARY_NAME = 'dictionary.txt'
MATRICES_DIR_NAME = 'matrices'
EIGVALS_DIR_NAME = 'eigvals'
EIGVEC_DIR_NAME = 'eigvec'


class Hamiltonian(abc.ABC):
    @abc.abstractmethod
    def get_hamiltonian(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_label(self) -> str:
        pass


class HamiltionianDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        data_limit: t.Optional[int] = None,
        label_idx: t.Union[int, t.Tuple[int, int]] = 1,
        threshold: float = 1.e-5,
        eig_decomposition: bool = False,
        format: str = 'numpy'
    ):
        dic_path = os.path.join(data_dir, DICTIONARY_NAME)
        self.dictionary = self.load_dict(dic_path)
        self.data_dir = data_dir
        self.data_limit = data_limit
        self.label_idx = label_idx
        self.threshold = threshold
        self.eig_dec = eig_decomposition
        self.format = format

      
    def __len__(self) -> int:
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)


    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Tuple[t.Tuple[torch.Tensor, torch.Tensor], t.Optional[t.Tuple[torch.Tensor, torch.Tensor]]]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tensor = self.load_data(MATRICES_DIR_NAME, idx)
        tensor = torch.stack((tensor.real, tensor.imag), dim=0)

        if self.eig_dec:
            eigvals = self.load_data(EIGVALS_DIR_NAME, idx)
            eigvec = self.load_data(EIGVEC_DIR_NAME, idx)
            eig_dec = eigvals, eigvec
        else:
            eig_dec = torch.zeros((1, tensor.shape[1])), torch.zeros((tensor.shape[0], tensor.shape[1]))

        if type(self.label_idx) == int:
            label = abs(float(self.dictionary[idx][self.label_idx]))
            label = 1. if label > self.threshold else 0.
        elif type(self.label_idx) == tuple:
            label = float(self.dictionary[idx][self.label_idx[0]]) * float(self.dictionary[idx][self.label_idx[1]])
            label = 1. if label < -self.threshold else 0.
        else:
            raise ValueError("Wrong label_idx type")
        label = torch.tensor(label)
        label = label.unsqueeze(0)

        return (tensor, label), eig_dec


    def load_dict(self, filepath: str) -> t.List[t.List[str]]:
      
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()

        parsed_data = [row.rstrip("\n").split(', ') for row in data]

        return parsed_data
    

    def load_data(self, dir: str, idx: int):
        if self.format == 'numpy':
            data_path = os.path.join(self.data_dir, dir, self.dictionary[idx][0] + '.npy')
            data = np.load(data_path)
        elif self.format == 'csr':
            data_path = os.path.join(self.data_dir, dir, self.dictionary[idx][0] + '.npz')
            data = sparse.load_npz(data_path)
            data = data.toarray()
        else:
            raise ValueError("Wrong format")
        return torch.from_numpy(data).type(torch.complex64)

  
def generate_data(
    hamiltionian: Hamiltonian,
    param_list: t.List[t.Dict[str, t.Any]],
    directory: str,
    eig_decomposition: bool = False,
    format: str = 'numpy',
):
    for i, params in tqdm(enumerate(param_list), 'Generating data'):
        filename = 'data_' + str(i)
        model = hamiltionian(**params)
        matrix = model.get_hamiltonian()
        label = model.get_label()

        if eig_decomposition:
            eigvals, eigvec = np.linalg.eigh(matrix)
        else:
            eigvals, eigvec = None, None

        save_data(matrix, label, directory, filename, eigvals, eigvec, format)


def save_data(
    matrix: np.ndarray,
    label: str,
    root_dir: str,
    filename: str,
    eigvals: t.Optional[np.ndarray] = None,
    eigvec: t.Optional[np.ndarray] = None,
    format: str = 'numpy',
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