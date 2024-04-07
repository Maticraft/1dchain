import typing as t

import torch


PAULI_BLOCKS = {
    '1': torch.eye(2),
    'x': torch.tensor([[0, 1], [1, 0]]),
    'iy': torch.tensor([[0, 1], [-1, 0]]),
    'z': torch.tensor([[1, 0], [0, -1]]),
}


class BlockExtractor:
    @classmethod
    def extract_block_sequences(cls, x: torch.Tensor, pauli_block_names: t.List[str]):
        '''
        assumes:
          x.shape = (batch_size, 4, 4*seq_size)
          pauli_block_names = list of length block_params_num with names of respective pauli block pairs, e.g. ['1x', 'xx', 'yz', 'zz']
        returns:
          torch.Tensor of shape (batch_size, block_params_num, seq_size)
        '''
        return torch.stack([cls._extract_sequence(x, *get_block_pair(pair_name)) for pair_name in pauli_block_names], dim=1)

    @staticmethod
    def _extract_sequence(x: torch.Tensor, block_a: torch.Tensor, block_b: torch.Tensor):
        '''
        assumes:
          x.shape = (batch_size, 4, 4*seq_size)
          block_a.shape = (2, 2)
          block_b.shape = (2, 2)
        returns:
          torch.Tensor of shape (batch_size, seq_size)
        '''
        seq_size = x.shape[-1] // 4
        block = torch.kron(block_a.to(x.device), block_b.to(x.device))
        block_expanded = block.unsqueeze(0).expand(x.shape[0], -1, -1).repeat(1, 1, seq_size)
        masked_block = x * block_expanded
        masked_block_reshaped = masked_block.transpose(-1, -2).reshape(-1, seq_size, 4, 4)
        return masked_block_reshaped.sum(dim=(-2, -1)) / block.abs().sum()


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
        blocks = [cls._block_generator(block_sequence, *get_block_pair(pair_name)) for block_sequence, pair_name in zip(block_params_sequence, pauli_block_names)]
        return sum(blocks)
    
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


def get_block_pair(pair_name: str):
    '''
    assumes:
      pair_name = eg. '1x', 'xx', 'iyz', 'zz'
    returns:
      block_a, block_b
    '''
    block_a_name, block_b_name = parse_string_to_pair(pair_name)
    return PAULI_BLOCKS[block_a_name], PAULI_BLOCKS[block_b_name]
    

def parse_string_to_pair(string: str):
    if not 'i' in string:
        return string[0], string[1]
    if string[0] == 'i':
        return string[:2], string[2:]
    return string[0], string[1:]


def get_matrix_from_strips(strips: torch.Tensor, sites_num: int, block_size: int = 4):
    matrix = torch.zeros((strips.shape[0], 2, sites_num*block_size, sites_num*block_size)).to(strips.device)
    strips_split = torch.tensor_split(strips, strips.shape[1] // 2, dim=1)
    for i, strip in enumerate(strips_split):
        offset = i - (len(strips_split) // 2)
        matrix_off = abs(offset)*block_size
        for j in range(sites_num):
            idx0 =  j*block_size
            idx1 = (j+1)*block_size
            if offset >= 0:
                matrix_idx0 = (idx0 + matrix_off) % (sites_num*block_size)
                matrix_idx1 = matrix_idx0 + block_size
            else:
                matrix_idx0 = (idx0 - matrix_off) % (sites_num*block_size)
                matrix_idx1 = matrix_idx0 + block_size
            matrix[:, :, idx0: idx1, matrix_idx0: matrix_idx1] = strip[:, :, :, idx0: idx1]
    return matrix


def get_strip(x: torch.Tensor, offset: int, sites_num: int, fill_mode: str = 'zeros', block_size: int = 4):
    strip = torch.zeros((x.shape[0], x.shape[1], block_size, sites_num*block_size)).to(x.device)
    x_off = abs(offset)*block_size
    for i in range(sites_num):
        idx0 =  i*block_size
        idx1 = idx0 + block_size
        if offset >= 0:
            idx0_off = (idx0 + x_off) % (sites_num * block_size)
            idx1_off = idx0_off + block_size
        else:
            idx0_off = (idx0 - x_off) % (sites_num * block_size)
            idx1_off = idx0_off + block_size
        strip[:, :, :, idx0: idx1] = x[:, :, idx0: idx1, idx0_off: idx1_off]
    if fill_mode == 'zeros':
        if offset > 0:
            strip[:, :, :, -x_off:] = 0.
        elif offset < 0:
            strip[:, :, :, :x_off] = 0.
        return strip
    elif fill_mode == 'circular':
        if offset > 0:
            strip[:, :, :, -x_off:] = strip[:, :, :, :x_off]
        elif offset < 0:
            strip[:, :, :, :x_off] = strip[:, :, :, -x_off:]
        return strip
    elif fill_mode == 'hamiltonian':
        return strip
    else:
        raise ValueError(f'Fill mode: {fill_mode} not implemented')