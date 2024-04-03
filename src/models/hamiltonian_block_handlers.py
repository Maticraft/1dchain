from itertools import product
import typing as t

import torch.nn as nn
import torch


PAULI_BLOCKS ={
    '1': torch.eye(2),
    'x': torch.tensor([[0, 1], [1, 0]]),
    'iy': torch.tensor([[0, 1], [-1, 0]]),
    'z': torch.tensor([[1, 0], [0, -1]]),
}


class BlockExtractor:
    pass


class BlockConstructor:
    @classmethod
    def generate_block_sequence(cls, block_params_sequence: torch.Tensor, pauli_block_names: t.List[str]):
        '''
        assumes:
          block_params_sequence.shape = (block_params_num, batch_size, seq_size)
          pauli_block_names = list of length block_params_num with names of respective pauli block pairs, e.g. ['1x', 'xx', 'yz', 'zz']
        returns:
          torch.Tensor of shape (batch_size, 4, 4*seq_size)
        '''
        blocks = [cls._block_generator(block_sequence, *cls._get_block_pair(pair_name)) for block_sequence, pair_name in zip(block_params_sequence,pauli_block_names)]
        return sum(blocks)
    
    @classmethod
    def _get_block_pair(cls, pair_name: str):
        '''
        assumes:
          pair_name = eg. '1x', 'xx', 'iyz', 'zz'
        returns:
          block_a, block_b
        '''
        block_a_name, block_b_name = cls._parse_string_to_pair(pair_name)
        return PAULI_BLOCKS[block_a_name], PAULI_BLOCKS[block_b_name]
    
    @staticmethod
    def _parse_string_to_pair(string: str):
        if not 'i' in string:
            return string[0], string[1]
        if string[0] == 'i':
            return string[:2], string[2:]
        return string[0], string[1:]

    @staticmethod
    def _block_generator(x: torch.Tensor, block_a: torch.Tensor, block_b: torch.Tensor):
        '''
        assumes:
          x.shape = (batch_size, seq_size)
          block_a.shape = (2, 2)
          block_b.shape = (2, 2)
        returns:
          torch.Tensor of shape (batch_size, 4, 4*seq_size)
        '''
        x_broadcast = x.unsqueeze(-1).unsqueeze(-1)
        block = torch.kron(block_a.to(x.device), block_b.to(x.device))
        block_expanded = block.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
        full_block = x_broadcast*block_expanded
        return torch.cat([full_block[:, i, :, :] for i in range(full_block.shape[1])], dim=-1)