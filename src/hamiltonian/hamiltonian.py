import abc
import typing as t
from enum import Enum
from functools import wraps

import numpy as np
import torch


def transform_default_representation_to_second_quantized_plus_minus_up_down(hamiltonian: np.ndarray) -> np.ndarray:
    '''
    place holder for the mapping (default -> second_quantized_plus_minus_up_down)
    as the default representation is the second_quantized_plus_minus_up_down no transformation is required
    '''
    return hamiltonian


def transform_default_representation_to_majorana_plus_minus_up_down(hamiltonian: np.ndarray) -> np.ndarray:
    '''
    transform the default representation to the majorana_plus_minus_up_down representation
    '''
    T_gamma_a = np.array(
        [
            # a_minus_up, a_minus_down, a_plus_up, a_plus_down
            [1., 0, 1., 0],         # gamma_plus_up
            [0, 1., 0, 1.],         # gamma_plus_down
            [1.j, 0, -1.j, 0],      # gamma_minus_up
            [0, 1.j, 0, -1.j]       # gamma_minus_down
        ]
    )
    T_gamma_a_plus = np.array(
        [
            # a_plus_up, a_plus_down, a_minus_up, a_minus_down
            [1., 0, 1., 0],         # gamma_plus_up
            [0, 1., 0, 1.],         # gamma_plus_down
            [-1.j, 0, 1.j, 0],      # gamma_minus_up
            [0, -1.j, 0, 1.j]       # gamma_minus_down
        ]
    )
    T_a_gamma = 0.5*np.array(
        [
            # gamma_plus_up, gamma_plus_down, gamma_minus_up, gamma_minus_down
            [1., 0, -1.j, 0],       # a_minus_up
            [0, 1., 0, -1.j],       # a_minus_down
            [1., 0, 1.j, 0],        # a_plus_up
            [0, 1., 0, 1.j]         # a_plus_down
        ]
    )

    # For each 4 x 4 block in the Hamiltonian, apply the transformation T_gamma_a_plus @ block @ T_a_gamma_minus
    hamiltonian_majorana = np.zeros_like(hamiltonian)
    for i in range(0, hamiltonian.shape[-2], 4):
        for j in range(0, hamiltonian.shape[-1], 4):
            block = hamiltonian[..., i:i+4, j:j+4].copy()
            # if i == j:
            #     block_majorana = T_gamma_a_plus @ block @ T_a_gamma
            # else:
            #     block_majorana = T_gamma_a_plus @ block @ T_a_gamma
            #     block_majorana = block_majorana.conj()
            block_majorana = T_gamma_a @ block @ T_a_gamma
            # reconstructed_block = np.triu(block_majorana, k = 0) - np.triu(block_majorana, k = 1).T # does not work for t blocks (actually it work if conjugate is eliminated)
            hamiltonian_majorana[..., i:i+4, j:j+4] = block_majorana
    return hamiltonian_majorana


def transform_majorana_plus_minus_up_down_representation_to_default(hamiltonian: np.ndarray) -> np.ndarray:
    '''
    transform the default representation to the majorana_plus_minus_up_down representation
    '''
    T_gamma_a = np.array(
        [
            # a_minus_up, a_minus_down, a_plus_up, a_plus_down
            [1., 0, 1., 0],         # gamma_plus_up
            [0, 1., 0, 1.],         # gamma_plus_down
            [1.j, 0, -1.j, 0],      # gamma_minus_up
            [0, 1.j, 0, -1.j]       # gamma_minus_down
        ]
    )
    T_a_gamma = 0.5*np.array(
        [
            # gamma_plus_up, gamma_plus_down, gamma_minus_up, gamma_minus_down
            [1., 0, -1.j, 0],       # a_minus_up
            [0, 1., 0, -1.j],       # a_minus_down
            [1., 0, 1.j, 0],        # a_plus_up
            [0, 1., 0, 1.j]         # a_plus_down
        ]
    )

    # For each 4 x 4 block in the Hamiltonian, apply the transformation T_gamma_a_plus @ block @ T_a_gamma_minus
    hamiltonian_majorana = np.zeros_like(hamiltonian)
    for i in range(0, hamiltonian.shape[-2], 4):
        for j in range(0, hamiltonian.shape[-1], 4):
            block = hamiltonian[..., i:i+4, j:j+4].copy()
            block_majorana = T_a_gamma @ block @ T_gamma_a
            hamiltonian_majorana[..., i:i+4, j:j+4] = block_majorana
    return hamiltonian_majorana


class RepresentationMapping(str, Enum):
    second_quantized_plus_minus_up_down = 'second_quantized_plus_minus_up_down'  # [a_plus_up, a_plus_down, a_minus_up, a_minus_down]
    second_quantized_polarization = 'second_quantized_polarization'  # only for tests for majorana polarization [a_plus_up, a_plus_down, -a_minus_down, a_minus_up]
    majorana_plus_minus_up_down = 'majorana_plus_minus_up_down'  # [gamma_plus_up, gamma_plus_down, gamma_minus_up, gamma_minus_down]
    inverse_majorana_plus_minus_up_down = 'inverse_majorana_plus_minus_up_down'  # creating default representation from majorana_plus_minus_up_down
    none = 'none'

    default = second_quantized_plus_minus_up_down

    def __str__(self):
        return self.value


REPRESENTATION_MAPPING_FUNCTION = {
    RepresentationMapping.second_quantized_plus_minus_up_down: transform_default_representation_to_second_quantized_plus_minus_up_down,
    RepresentationMapping.majorana_plus_minus_up_down: transform_default_representation_to_majorana_plus_minus_up_down,
    RepresentationMapping.second_quantized_polarization: transform_default_representation_to_second_quantized_plus_minus_up_down,
    RepresentationMapping.inverse_majorana_plus_minus_up_down: transform_majorana_plus_minus_up_down_representation_to_default,
    RepresentationMapping.none: lambda x: x
}


class Hamiltonian(abc.ABC):        
    @abc.abstractmethod
    def get_hamiltonian_matrix(self) -> np.ndarray:
        pass

    def get_hamiltonian(self, representation_mapping: RepresentationMapping = RepresentationMapping.none) -> np.ndarray:
        hamiltonian = self.get_hamiltonian_matrix()
        hamiltonian_in_representation = REPRESENTATION_MAPPING_FUNCTION[representation_mapping](hamiltonian)
        return hamiltonian_in_representation

    def get_hamiltonian_tensor(self, representation_mapping: RepresentationMapping = RepresentationMapping.none) -> torch.Tensor:
        hamiltonian = self.get_hamiltonian(representation_mapping)
        complex_hamiltonian = torch.from_numpy(hamiltonian).type(torch.complex64)
        return torch.stack((complex_hamiltonian.real, complex_hamiltonian.imag), dim=0)

    @abc.abstractmethod
    def get_label(self) -> str:
        pass

    @abc.abstractmethod
    def set_parameter(self, parameter_name: str, value: t.Any):
        pass
    

class TestHamiltonian(Hamiltonian):
    def __init__(self):
        self.h = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1]
            ],
            dtype=np.complex128
        )

    def get_hamiltonian_matrix(self) -> np.ndarray:
        return self.h
    
    def get_label(self) -> str:
        return 'Test Hamiltonian'
    
    def set_parameter(self, parameter_name: str, value: t.Any):
        raise NotImplementedError('Not implemented')


REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR = {
    'potential': 'z1',
    'magnetic_field': 'zz',
    'delta': 'iyiy',
    'spin_twist': 'zx',
    'hopping_same': 'z1',
    'hopping_spin_flip': '1iy'
}
IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR = {
    'potential': 'z1',
    'magnetic_field': 'zz',
    'delta': 'xiy',
    'spin_twist': '1iy',
    'hopping_same': 'zz',
    'hopping_same_qd': '1z',
    'hopping_spin_flip': 'zx',
    'hopping_spin_flip_qd': '1x',
}
    