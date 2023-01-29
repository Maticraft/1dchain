import abc
import os
import typing as t

import numpy as np
from torch.utils.data import Dataset
import torch
from tqdm import tqdm


class Hamiltonian(abc.ABC):
    @abc.abstractmethod
    def get_hamiltonian(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_label(self) -> int:
        pass


class HamiltionianDataset(Dataset):

    def __init__(
        self,
        dictionary: str,
        root_dir: str,
        data_limit: t.Optional[int] = None,
        label_idx: t.Union[int, t.Tuple[int, int]] = 1,
        threshold: float = 1.e-5,
    ):
        self.dictionary = self.load_dict(dictionary)
        self.root_dir = root_dir
        self.data_limit = data_limit
        self.label_idx = label_idx
        self.threshold = threshold

      
    def __len__(self) -> int:
        if self.data_limit != None:
            return self.data_limit
        else:
            return len(self.dictionary)


    def __getitem__(self, idx: t.Union[int, torch.Tensor]) -> t.Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        matrix_name = os.path.join(self.root_dir, self.dictionary[idx][0] + '.npy')
        matrix = np.load(matrix_name)
        matrix_r = np.real(matrix)
        matrix_im = np.imag(matrix)

        tensor = torch.from_numpy(np.stack((matrix_r, matrix_im), axis=0)).float()

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

        return (tensor, label)


    def load_dict(self, filepath: str) -> t.List[t.List[str]]:
      
        with open(filepath, 'r') as dictionary:
            data = dictionary.readlines()

        parsed_data = [row.rstrip("\n").split(', ') for row in data]

        return parsed_data

  
def generate_data(hamiltionian: Hamiltonian, param_list: t.List[t.Dict[str, t.Any]], directory: str):
    for i, params in tqdm(enumerate(param_list), 'Generating data'):
        model = hamiltionian(**params)
        matrix = model.get_hamiltonian()
        label = model.get_label()
        filename = 'data_' + str(i)
        save_data(matrix, label, directory, filename)


def save_data(matrix: np.ndarray, label: int, root_dir: str, filename: str):
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    matrix_dir = os.path.join(root_dir, 'matrices')
    if not os.path.isdir(matrix_dir):
        os.makedirs(matrix_dir)
    matrix_name = os.path.join(matrix_dir, filename + '.npy')

    np.save(matrix_name, matrix)

    with open(os.path.join(root_dir, 'dictionary.txt'), 'a') as dictionary:
        dictionary.write(filename + ', ' + str(label) + '\n')