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
    def generate_block_matrix(cls, real_block_params: torch.Tensor, imag_block_params: torch.Tensor, real_pauli_block_names: t.List[str], imag_pauli_block_names: t.List[str], offset: int = 0):
        '''
        assumes:
          block_params.shape = (batch_size, block_params_num, seq_size)
          pauli_block_names = list of length block_params_num with names of respective pauli block pairs, e.g. ['1x', 'xx', 'yz', 'zz']
        returns:
          torch.Tensor of shape (batch_size, 4, 4*seq_size)
        '''
        if offset == 0:
            block_sequence = cls._construct_on_site_blocks(real_block_params, imag_block_params, real_pauli_block_names, imag_pauli_block_names)
        else:
            upper_interaction_strip, lower_interaction_strip = cls._construct_interactions_blocks(real_block_params, imag_block_params, real_pauli_block_names, imag_pauli_block_names, offset - 1)
            block_sequence = torch.cat([lower_interaction_strip, torch.zeros_like(lower_interaction_strip), upper_interaction_strip], dim=1)
        block_matrix = get_matrix_from_strips(block_sequence, sites_num=block_sequence.shape[-1] // 4)
        return block_matrix
        

    @classmethod
    def generate_block_sequence(cls, block_params_sequence: torch.Tensor, pauli_block_names: t.List[str], lower_block_transpose: bool = False):
        '''
        assumes:
          block_params_sequence.shape = (block_params_num, batch_size, seq_size)
          pauli_block_names = list of length block_params_num with names of respective pauli block pairs, e.g. ['1x', 'xx', 'yz', 'zz']
        returns:
          torch.Tensor of shape (batch_size, 4, 4*seq_size)
        '''
        blocks = [cls._block_generator(block_sequence, *get_block_pair(pair_name), lower_block_transpose) for block_sequence, pair_name in zip(block_params_sequence, pauli_block_names)]
        if len(blocks) > 0:
            return sum(blocks)
        return torch.zeros((block_params_sequence.shape[1], 4, 4*block_params_sequence.shape[2])).to(block_params_sequence.device)
    
    @staticmethod
    def _block_generator(x: torch.Tensor, block_a: torch.Tensor, block_b: torch.Tensor, lower_block_transpose: bool = False):
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
        if lower_block_transpose:
            block[2:, :2] = block[2:, :2].T.clone()
            block[2:, 2:] = block[2:, 2:].T.clone()
        block_expanded = block.unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
        full_block = x_broadcast*block_expanded
        return torch.cat([full_block[:, i, :, :] for i in range(full_block.shape[1])], dim=-1)
    
    @classmethod
    def _construct_interactions_blocks(cls, real_interactions: torch.Tensor, imag_interactions: torch.Tensor, real_pauli_block_names: t.List[str], imag_pauli_block_names: t.List[str], interaction_strip_idx: int = 0, block_size: int = 4):
        '''
        interactions.shape = (batch_size, interaction_params_per_strip, N)
        '''
        interaction_strip_real = cls.generate_block_sequence(real_interactions.transpose(0, 1), real_pauli_block_names)
        interaction_strip_imag = cls.generate_block_sequence(imag_interactions.transpose(0, 1), imag_pauli_block_names)
        upper_interaction_strip = torch.stack([interaction_strip_real, interaction_strip_imag], dim=1) # of shape (batch_size, 2, 4, 4*N)
        periodic_interactions_offset = (interaction_strip_idx + 1) * block_size
        lower_interaction_strip = cls._generate_lower_strip_from_upper_strip(upper_interaction_strip, periodic_interactions_offset)
        return upper_interaction_strip, lower_interaction_strip
    
    @staticmethod
    def _generate_lower_strip_from_upper_strip(upper_strip: torch.Tensor, periodic_interactions_offset: int):
        # shift the periodic interactions from the right side of the upper strip to the left side of the lower strip
        lower_strip = torch.cat([upper_strip[:, :, :, -periodic_interactions_offset:], upper_strip[:, :, :, :-periodic_interactions_offset]], dim=-1)
        lower_strip[:, 1::2] *= -1 # conjugation of imaginary part
        # transpose the strip (switch the order of the non-diagonal interactions in the lower strip)
        right_upper_coeff = lower_strip[:, :, 0, 1::4].clone()
        left_upper_coeff = lower_strip[:, :, 1, 0::4].clone()
        lower_strip[:, :, 0, 1::4] = left_upper_coeff
        lower_strip[:, :, 1, 0::4] = right_upper_coeff

        right_lower_coeff = lower_strip[:, :, 2, 3::4].clone()
        left_lower_coeff = lower_strip[:, :, 3, 2::4].clone()
        lower_strip[:, :, 2, 3::4] = left_lower_coeff
        lower_strip[:, :, 3, 2::4] = right_lower_coeff
        return lower_strip
    
    @classmethod
    def _construct_on_site_blocks(cls, real_on_site_blocks: torch.Tensor, imag_on_site_blocks: torch.Tensor, real_pauli_block_names: t.List[str], imag_pauli_block_names: t.List[str],):
        '''
        assumes on_site_blocks.shape = (batch_size, total_on_site_params, seq_size)
        '''
        real_blocks = cls.generate_block_sequence(real_on_site_blocks.transpose(0, 1), real_pauli_block_names)
        imaginary_blocks = cls.generate_block_sequence(imag_on_site_blocks.transpose(0, 1), imag_pauli_block_names)
        return torch.stack([real_blocks, imaginary_blocks], dim=1)


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


def get_strip(x: torch.Tensor, offset: int, fill_mode: str = 'zeros', block_size: int = 4):
    sites_num = x.shape[-1] // block_size
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
    