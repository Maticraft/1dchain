import typing as t


from dataclasses import dataclass
from dataclasses_json import dataclass_json
import json_fix
import torch
import torch.nn as nn

from src.hamiltonian.hamiltonian import REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR, IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR

PAULI_BLOCKS = {
    '1': torch.eye(2),
    'x': torch.tensor([[0, 1], [1, 0]]),
    'iy': torch.tensor([[0, 1], [-1, 0]]),
    'z': torch.tensor([[1, 0], [0, -1]]),
}
ALL_PAIRS = ['11', '1x', '1iy', '1z', 'x1', 'xx', 'xiy', 'xz', 'iy1', 'iyx', 'iyiy', 'iyz', 'z1', 'zx', 'ziy', 'zz']

@dataclass_json
@dataclass
class HamiltonianParams:
    on_site_real_params: t.Tuple[str] = "potential", "magnetic_field", "delta"
    on_site_imag_params: t.Tuple[str] = "delta",
    inter_site_real_params: t.Tuple[str] = "hopping_same", "hopping_spin_flip"
    inter_site_imag_params: t.Tuple[str] = "hopping_spin_flip_qd",

    def __json__(self):
        return self.to_dict()


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
        if not pauli_block_names:
            return torch.zeros((x.shape[0], 0, x.shape[-1] // 4)).to(x.device)
        return torch.stack([cls._extract_sequence(x, *get_block_pair(pair_name)) for pair_name in pauli_block_names], dim=1)

    @staticmethod
    def _extract_sequence(x: torch.Tensor, blocks_a: t.List[torch.Tensor], blocks_b: t.List[torch.Tensor]):
        '''
        assumes:
          x.shape = (batch_size, 4, 4*seq_size)
          block_a.shape = (2, 2)
          block_b.shape = (2, 2)
        returns:
          torch.Tensor of shape (batch_size, seq_size)
        '''
        seq_size = x.shape[-1] // 4
        block = torch.stack([torch.kron(block_a.to(x.device), block_b.to(x.device)) for block_a, block_b in zip(blocks_a, blocks_b)], dim=0).sum(dim=0)
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
    def _block_generator(x: torch.Tensor, blocks_a: t.List[torch.Tensor], blocks_b: t.List[torch.Tensor], lower_block_transpose: bool = False):
        '''
        assumes:
          x.shape = (batch_size, seq_size)
          block_a.shape = (2, 2)
          block_b.shape = (2, 2)
        returns:
          torch.Tensor of shape (batch_size, 4, 4*seq_size)
        '''
        x_broadcast = x.unsqueeze(-1).unsqueeze(-1)
        block = torch.stack([torch.kron(block_a.to(x.device), block_b.to(x.device)) for block_a, block_b in zip(blocks_a, blocks_b)], dim=0).sum(dim=0)
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
    

class HamiltonianConstructor(nn.Module):
    def __init__(
        self,
        hamiltonian_params: HamiltonianParams | t.Dict[str, t.Any],
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        real_param_to_block: t.Dict[str, str] = REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR,
        imag_param_to_block: t.Dict[str, str] = IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR,
    ):
        super(HamiltonianConstructor, self).__init__()

        self.real_param_to_block = real_param_to_block
        self.imag_param_to_block = imag_param_to_block

        if isinstance(hamiltonian_params, dict):
            hamiltonian_params = HamiltonianParams(**hamiltonian_params)
        self.hamiltonian_params = hamiltonian_params

        self.on_site_real_block_names = [self.real_param_to_block[param] for param in self.hamiltonian_params.on_site_real_params]
        self.on_site_imag_block_names = [self.imag_param_to_block[param] for param in self.hamiltonian_params.on_site_imag_params]
        
        self.inter_site_real_block_names = [self.real_param_to_block[param] for param in self.hamiltonian_params.inter_site_real_params]
        self.inter_site_imag_block_names = [self.imag_param_to_block[param] for param in self.hamiltonian_params.inter_site_imag_params]
        
        self.min_inter_site_interaction_range = min_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.
        self.max_inter_site_interaction_range = max_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.

    @property
    def num_on_site_real_params(self):
        return len(self.on_site_real_block_names)
    
    @property
    def num_on_site_imag_params(self):
        return len(self.on_site_imag_block_names)
    
    @property
    def num_on_site_params(self):
        return self.num_on_site_real_params + self.num_on_site_imag_params
    
    @property
    def num_inter_site_real_params(self):
        return len(self.inter_site_real_block_names)
    
    @property
    def num_inter_site_imag_params(self):
        return len(self.inter_site_imag_block_names)
    
    @property
    def num_inter_site_params(self):
        return self.num_inter_site_real_params + self.num_inter_site_imag_params
    
    def get_params_map(self, on_site_params: torch.Tensor, inter_site_params: torch.Tensor):
        '''
        Args:
          on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        Returns:
          params_map: dict
        '''
        params_map = {
            'on_site_real_params': {
                param_name: on_site_params[..., param_idx, :]
                for param_idx, param_name in enumerate(self.hamiltonian_params.on_site_real_params)
            },
            'on_site_imag_params': {
                param_name: on_site_params[..., self.num_on_site_real_params + param_idx, :]
                for param_idx, param_name in enumerate(self.hamiltonian_params.on_site_imag_params)
            },
            'inter_site_real_params': {
                param_name: inter_site_params[..., param_idx, :]
                for param_idx, param_name in enumerate(self.hamiltonian_params.inter_site_real_params)
            },
            'inter_site_imag_params': {
                param_name: inter_site_params[..., self.num_inter_site_real_params + param_idx, :]
                for param_idx, param_name in enumerate(self.hamiltonian_params.inter_site_imag_params)
            }
        }
        return params_map


    def forward(self, on_site_params: torch.Tensor, inter_site_params: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        Returns:
          hamiltonian matrix: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
        '''
        hamiltonian = self._construct_hamiltonian_matrix(on_site_params, inter_site_params)
        return hamiltonian
    
    def generate_random_hamiltonian(self, num_samples: int, seq_size: int, device: t.Optional[torch.device] = None, site_constant: bool = False) -> torch.Tensor:
        '''
        Args:
          num_samples: int
          seq_size: int
          device: t.Optional[torch.device]
        Returns:
          hamiltonian: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
        '''
        inter_site_interaction_range = self.max_inter_site_interaction_range - self.min_inter_site_interaction_range
        if site_constant:
            on_site_params = torch.randn(num_samples, self.num_on_site_params, 1, device=device).expand(-1, -1, seq_size)
            inter_site_params = torch.randn(num_samples, inter_site_interaction_range, self.num_inter_site_params, 1, device=device).expand(-1, -1, -1, seq_size)
        else:  
            on_site_params = torch.randn(num_samples, self.num_on_site_params, seq_size, device=device)
            inter_site_params = torch.randn(num_samples, inter_site_interaction_range, self.num_inter_site_params, seq_size, device=device)
        return self._construct_hamiltonian_matrix(on_site_params, inter_site_params)
    
    def _construct_hamiltonian_matrix(self, on_site_params: torch.Tensor, inter_site_params: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        Returns:
          hamiltonian_matrix: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
        '''
        block_size = 4
        flattened_on_site_params = on_site_params.flatten(start_dim=0, end_dim=-3)
        flattened_on_site_real_params = flattened_on_site_params[:, :self.num_on_site_real_params]
        flattened_on_site_imag_params = flattened_on_site_params[:, self.num_on_site_real_params:]
        
        on_site_block_seq = torch.stack((
            BlockConstructor.generate_block_sequence(flattened_on_site_real_params.transpose(0, 1), self.on_site_real_block_names, lower_block_transpose=False), # real part
            BlockConstructor.generate_block_sequence(flattened_on_site_imag_params.transpose(0, 1), self.on_site_imag_block_names, lower_block_transpose=False) # imaginary part
        ), dim=-3)

        flattened_inter_site_params = inter_site_params.flatten(start_dim=0, end_dim=-4)
        flattened_inter_site_real_params = flattened_inter_site_params[:, :, :self.num_inter_site_real_params]
        flattened_inter_site_imag_params = flattened_inter_site_params[:, :, self.num_inter_site_real_params:]
        inter_site_block_seq = torch.cat([
            torch.stack((
                BlockConstructor.generate_block_sequence(flattened_inter_site_real_params[:, inter_site_interaction_range_idx].transpose(0, 1), self.inter_site_real_block_names), # real part
                BlockConstructor.generate_block_sequence(flattened_inter_site_imag_params[:, inter_site_interaction_range_idx].transpose(0, 1), self.inter_site_imag_block_names) # imaginary part
            ), dim=-3)
            for inter_site_interaction_range_idx in range(flattened_inter_site_params.shape[-3])
        ], dim=-3)

        skipped_inter_site_block_seq = [
            torch.zeros(flattened_inter_site_params.shape[0], 2, block_size, inter_site_block_seq.shape[-1], device=flattened_inter_site_params.device)
            for _ in range(self.min_inter_site_interaction_range - 1)
        ] # why is it not working for min_inter_site_interaction_range >=2?
        if len(skipped_inter_site_block_seq) > 0:
            inter_site_block_seq = torch.cat([*skipped_inter_site_block_seq, inter_site_block_seq], dim=-3)

        strips = torch.cat([torch.zeros_like(inter_site_block_seq), on_site_block_seq, inter_site_block_seq], dim=-3)
        hamiltonian_matrix = get_matrix_from_strips(strips, on_site_params.shape[-1], block_size)
        periodic_elements = torch.tril(hamiltonian_matrix, diagonal=-block_size).transpose(-2, -1) # because of periodic inter site elements
        upper_triangular_matrix = torch.triu(hamiltonian_matrix) + torch.stack((periodic_elements[:, 0], -periodic_elements[:, 1]), dim=1)
        lower_triangular_matrix = torch.triu(upper_triangular_matrix, diagonal=1)
        lower_triangular_matrix = torch.stack((lower_triangular_matrix[:, 0], -lower_triangular_matrix[:, 1]), dim=1).transpose(-2, -1)
        hamiltonian_matrix_hermitian = upper_triangular_matrix + lower_triangular_matrix
        hamiltonian_matrix_unflattened = torch.unflatten(hamiltonian_matrix_hermitian, dim=0, sizes=on_site_params.shape[:-2])
        return hamiltonian_matrix_unflattened


class HamiltonianExtractor(nn.Module):
    def __init__(
        self,
        hamiltonian_params: HamiltonianParams | t.Dict[str, t.Any],
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        real_param_to_block: t.Dict[str, str] = REAL_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR,
        imag_param_to_block: t.Dict[str, str] = IMAG_HAMILTONIAN_PROPERTY_TO_BLOCK_PAIR,
    ):
        super(HamiltonianExtractor, self).__init__()
        if isinstance(hamiltonian_params, dict):
            hamiltonian_params = HamiltonianParams(**hamiltonian_params)
        self.hamiltonian_params = hamiltonian_params

        self.real_param_to_block = real_param_to_block
        self.imag_param_to_block = imag_param_to_block

        self.on_site_real_block_names = [self.real_param_to_block[param] for param in self.hamiltonian_params.on_site_real_params]
        self.on_site_imag_block_names = [self.imag_param_to_block[param] for param in self.hamiltonian_params.on_site_imag_params]
        
        self.inter_site_real_block_names = [self.real_param_to_block[param] for param in self.hamiltonian_params.inter_site_real_params]
        self.inter_site_imag_block_names = [self.imag_param_to_block[param] for param in self.hamiltonian_params.inter_site_imag_params]

        self.min_inter_site_interaction_range = min_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.
        self.max_inter_site_interaction_range = max_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.

    @property
    def num_on_site_real_params(self):
        return len(self.on_site_real_block_names)
    
    @property
    def num_on_site_imag_params(self):
        return len(self.on_site_imag_block_names)
    
    @property
    def num_on_site_params(self):
        return self.num_on_site_real_params + self.num_on_site_imag_params
    
    @property
    def num_inter_site_real_params(self):
        return len(self.inter_site_real_block_names)
    
    @property
    def num_inter_site_imag_params(self):
        return len(self.inter_site_imag_block_names)
    
    @property
    def num_inter_site_params(self):
        return self.num_inter_site_real_params + self.num_inter_site_imag_params


    def forward(self, hamiltonian: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
          hamiltonian matrix: torch.Tensor with shape (batch_size, 2, 4 x seq_size, 4 x seq_size)     
        Returns:
          Tuple:
          :on_site_params: torch.Tensor with shape (batch_size, num_on_site_params, seq_size)
          :inter_site_params: torch.Tensor with shape (batch_size, inter_site_interaction_range, num_inter_site_params, seq_size)
        '''
        on_site_params, inter_site_params = self._extract_params_from_hamiltonian_matrix(hamiltonian)
        return on_site_params, inter_site_params
    
    def from_params_map(self, params_map: t.Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Args:
          params_map: dict
        Returns:
          Tuple:
          :on_site_params: torch.Tensor with shape (batch_size, num_on_site_params, seq_size)
          :inter_site_params: torch.Tensor with shape (batch_size, inter_site_interaction_range, num_inter_site_params, seq_size)
        '''
        on_site_real_params = [params_map['on_site_real_params'][param_name] for param_name in self.hamiltonian_params.on_site_real_params]
        on_site_imag_params = [params_map['on_site_imag_params'][param_name] for param_name in self.hamiltonian_params.on_site_imag_params]
        on_site_params = torch.stack([*on_site_real_params, *on_site_imag_params], dim=-2)

        inter_site_real_params = [params_map['inter_site_real_params'][param_name] for param_name in self.hamiltonian_params.inter_site_real_params]
        inter_site_imag_params = [params_map['inter_site_imag_params'][param_name] for param_name in self.hamiltonian_params.inter_site_imag_params]
        inter_site_params = torch.stack([*inter_site_real_params, *inter_site_imag_params], dim=-2)
        return on_site_params, inter_site_params


    def _extract_params_from_hamiltonian_matrix(self, hamiltonian: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
          hamiltonian matrix: torch.Tensor with shape (batch_size, 2, 4 x seq_size, 4 x seq_size)     
        Returns:
          Tuple:
          :on_site_params: torch.Tensor with shape (batch_size, num_on_site_params, seq_size)
          :inter_site_params: torch.Tensor with shape (batch_size, inter_site_interaction_range, num_inter_site_params, seq_size)
        '''
        block_size = 4
        on_site_strip = get_strip(hamiltonian, 0, 'hamiltonian', block_size)
        on_site_real_params = BlockExtractor.extract_block_sequences(on_site_strip[:, 0], self.on_site_real_block_names)
        on_site_imag_params = BlockExtractor.extract_block_sequences(on_site_strip[:, 1], self.on_site_imag_block_names)
        on_site_params = torch.cat([on_site_real_params, on_site_imag_params], dim=-2)

        inter_site_strips = torch.stack([
            get_strip(hamiltonian, strip_idx, 'hamiltonian', block_size)
            for strip_idx in range(self.min_inter_site_interaction_range, self.max_inter_site_interaction_range)
        ], dim=1)
        inter_site_real_params = BlockExtractor.extract_block_sequences(inter_site_strips.flatten(0, 1)[:, 0], self.inter_site_real_block_names)
        inter_site_imag_params = BlockExtractor.extract_block_sequences(inter_site_strips.flatten(0, 1)[:, 1], self.inter_site_imag_block_names)
        inter_site_params = torch.cat([inter_site_real_params, inter_site_imag_params], dim=-2)
        inter_site_params = torch.unflatten(inter_site_params, dim=0, sizes=(inter_site_strips.shape[0], inter_site_strips.shape[1]))
        return on_site_params, inter_site_params


class HamiltonianConverter(nn.Module):
    def __init__(self, hamiltonian_params: HamiltonianParams | t.Dict[str, t.Any], **kwargs):
        super(HamiltonianConverter, self).__init__()
        if isinstance(hamiltonian_params, dict):
            hamiltonian_params = HamiltonianParams(**hamiltonian_params)
        self.hamiltonian_params = hamiltonian_params
        self.hamiltonian_constructor = HamiltonianConstructor(hamiltonian_params, **kwargs)
        self.hamiltonian_extractor = HamiltonianExtractor(hamiltonian_params, **kwargs)
    
    def from_matrix_to_params(self, hamiltonian: torch.Tensor) -> t.Dict[str, torch.Tensor]:
        '''
        Args:
          hamiltonian matrix: torch.Tensor with shape (batch_size, 2, 4 x seq_size, 4 x seq_size)     
        Returns:
          params_map: dict
        '''
        on_site_params, inter_site_params = self.hamiltonian_extractor(hamiltonian)
        return self.hamiltonian_constructor.get_params_map(on_site_params, inter_site_params)
    
    def from_params_to_matrix(self, hamiltonian_params: t.Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Args:
          hamiltonian_params: dict
        Returns:
            hamiltonian matrix: torch.Tensor with shape (batch_size, 2, 4 x seq_size, 4 x seq_size)     
        '''
        on_site_params, inter_site_params = self.hamiltonian_extractor.from_params_map(hamiltonian_params)
        return self.hamiltonian_constructor(on_site_params, inter_site_params)


def get_block_pair(pair_name: str):
    '''
    assumes:
      pair_name = eg. '1x 1z', 'xx', 'iyz', 'zz'
        if pair_name contains 'space' then all blocks withing the pair_name are summed
    returns:
      block_a, block_b
    '''
    block_a_name, block_b_name = parse_string_to_pair(pair_name)
    pauli_blocks_a = [PAULI_BLOCKS[block_name] for block_name in block_a_name]
    pauli_blocks_b = [PAULI_BLOCKS[block_name] for block_name in block_b_name]
    return pauli_blocks_a, pauli_blocks_b
    

def parse_string_to_pair(string: str):
    blocks_a = []
    blocks_b = []
    for block in string.split(' '):
        if not 'i' in block:
            blocks_a.append(block[0])
            blocks_b.append(block[1])
        elif block[0] == 'i':
            blocks_a.append(block[:2])
            blocks_b.append(block[2:])
        else:
            blocks_a.append(block[0])
            blocks_b.append(block[1:])
    return blocks_a, blocks_b


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
    