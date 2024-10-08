from abc import abstractmethod
import jsonpickle
import json_fix
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hamiltonian.hamiltonian_torch_handlers import BlockConstructor, get_matrix_from_strips
from src.models.base_models import MLP

class HiddenRepresentationGenerator(nn.Module):
    def __init__(self, hidden_representation_size: int):
        super(HiddenRepresentationGenerator, self).__init__()
        self.hidden_representation_size = hidden_representation_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x: torch.Tensor with shape (..., num_features)
        Returns:
          torch.Tensor with shape (..., hidden_representation_size, seq_size)
        '''
        raise NotImplementedError('HiddenRepresentationGenerator.forward not implemented')


class BaselineHiddenRepresentationGenerator(HiddenRepresentationGenerator):
    baseline_default_hidden_mlp_config = {
        'layers_num': 3,
        'input_size': 100,
        'hidden_size': 64,
        'output_size': 14,
        'activation': 'relu',
        'final_activation': 'none'
    }
    def __init__(self, num_hidden_mlps: int = 64, **hiden_mlp_config: t.Dict[str, t.Any]):
        super(BaselineHiddenRepresentationGenerator, self).__init__(num_hidden_mlps)
        self.hidden_mlp_config = hiden_mlp_config
        self.mlps = nn.ModuleList([MLP(**hiden_mlp_config) for _ in range(num_hidden_mlps)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x: torch.Tensor with shape (..., num_features)
        Returns:
          torch.Tensor with shape (..., num_hidden_mlps, mlp_output_size)
        '''
        hidden_representation = [mlp(x) for mlp in self.mlps]
        return torch.stack(hidden_representation, dim=-2)
    
    # make model json serializable
    def __json__(self):
        return {
            'num_hidden_mlps': self.hidden_representation_size,
            'hidden_mlp_config': self.hidden_mlp_config
        }


class MajoranaRepresentationHamiltonianGenerator(nn.Module):
    def __init__(
        self,
        hidden_representation_generator: HiddenRepresentationGenerator,
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        **kwargs: t.Any
    ):
        super(MajoranaRepresentationHamiltonianGenerator, self).__init__()
        self.on_site_block_names = ['iyiy', 'iy1', 'iyz', 'ziy']
        self.num_on_site_params = len(self.on_site_block_names)
        self.inter_site_block_names = ['1x', '1iy', '1z', 'iy1', 'iyx']
        # self.inter_site_signs = [-1, 1]
        self.num_inter_site_params = len(self.inter_site_block_names) 
        self.min_inter_site_interaction_range = min_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.
        self.max_inter_site_interaction_range = max_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.

        # NNs
        self.hidden_representation_generator: HiddenRepresentationGenerator = hidden_representation_generator
        self.hidden_channels = self.hidden_representation_generator.hidden_representation_size
        # self.on_site_converter = nn.Conv1d(self.hidden_channels, self.num_on_site_params, 1)
        self.inter_site_converters = nn.ModuleList([
            nn.Conv1d(self.hidden_channels, self.hidden_channels, kernel_size=2, stride=1, dilation=interaction_range)
            for interaction_range in range(self.min_inter_site_interaction_range, self.max_inter_site_interaction_range)
        ])
        self.inter_site_convs = nn.ConvTranspose2d(self.hidden_channels, 1, 4, stride=4)
        self.on_site_convs = nn.ConvTranspose2d(self.hidden_channels, 1, 4, stride=4)

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          x: torch.Tensor with shape (..., num_features)
        Returns:
          hamiltonian: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
        '''
        on_site_params, inter_site_params = self._nn_forward(x)
        hamiltonian = self._construct_hamiltonian_matrix(on_site_params, inter_site_params)
        return hamiltonian

    def _nn_forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
          x: torch.Tensor with shape (..., num_features)
        Returns:
          :on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          :inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        '''
        hidden_representation = self.hidden_representation_generator(x)
        # on_site_params = self.on_site_converter(hidden_representation)
        on_site_params = self.on_site_convs(hidden_representation.unsqueeze(-2))
        inter_site_params = torch.stack([
            converter(F.pad(hidden_representation, pad=(0, interaction_range), mode='circular')) for interaction_range, converter in zip(range(self.min_inter_site_interaction_range, self.max_inter_site_interaction_range), self.inter_site_converters)
        ], dim=-3) # (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        inter_site_params = torch.stack([self.inter_site_convs(inter_site_params[..., i, :, :].unsqueeze(-2)) for i in range(inter_site_params.shape[-3])], dim=-4)
        return on_site_params, inter_site_params
    
    def _construct_hamiltonian_matrix(self, on_site_params: torch.Tensor, inter_site_params: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        Returns:
          :hamiltonian matrix: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
        '''
        block_size = 4
        # flattened_on_site_params = on_site_params.flatten(start_dim=0, end_dim=-3)
        # on_site_block_seq = torch.stack((
        #     torch.zeros(flattened_on_site_params.shape[0], block_size, block_size*flattened_on_site_params.shape[-1], device=flattened_on_site_params.device), # real part is zeroed in majorana representation
        #     BlockConstructor.generate_block_sequence(flattened_on_site_params.transpose(0, 1), self.on_site_block_names, lower_block_transpose=False) # imaginary part
        # ), dim=-3)

        # flattened_inter_site_params = inter_site_params.flatten(start_dim=0, end_dim=-4)
        # # flattened_inter_site_params_expanded = flattened_inter_site_params.repeat(1, 1, len(self.inter_site_block_names), 1)
        # # flattened_inter_site_params_expanded = flattened_inter_site_params_expanded * torch.tensor(self.inter_site_signs).to(flattened_inter_site_params_expanded.device).unsqueeze(-1)
        # inter_site_block_seq = torch.cat([
        #     torch.stack((
        #         torch.zeros(flattened_inter_site_params.shape[0], block_size, block_size*flattened_inter_site_params.shape[-1], device=flattened_inter_site_params.device), # real part is zeroed in majorana representation
        #         BlockConstructor.generate_block_sequence(flattened_inter_site_params[:, inter_site_interaction_range_idx].transpose(0, 1), self.inter_site_block_names) # imaginary part
        #     ), dim=-3)
        #     for inter_site_interaction_range_idx in range(flattened_inter_site_params.shape[-3])
        # ], dim=-3)
        flattened_on_site_params = on_site_params.flatten(start_dim=0, end_dim=-4)
        on_site_block_seq = torch.cat((
            torch.zeros_like(flattened_on_site_params),
            flattened_on_site_params,
        ), dim=-3)
        
        flattened_inter_site_params = inter_site_params.flatten(start_dim=0, end_dim=-5)
        inter_site_block_seq = torch.cat((
            torch.zeros_like(flattened_inter_site_params),
            flattened_inter_site_params,
        ), dim=-3).squeeze(-4)

        skipped_inter_site_block_seq = [
            torch.zeros(flattened_inter_site_params.shape[0], 2, block_size, flattened_inter_site_params.shape[-1], device=flattened_inter_site_params.device) # mulitply shape -1 by block size
            for _ in range(self.min_inter_site_interaction_range - 1)
        ]
        if len(skipped_inter_site_block_seq) > 0:
            inter_site_block_seq = torch.cat([*skipped_inter_site_block_seq, inter_site_block_seq], dim=-3)

        strips = torch.cat([torch.zeros_like(inter_site_block_seq), on_site_block_seq, inter_site_block_seq], dim=-3)
        hamiltonian_matrix = get_matrix_from_strips(strips, int(on_site_params.shape[-1] / block_size), block_size)
        upper_triangular_matrix = torch.triu(hamiltonian_matrix) - torch.tril(hamiltonian_matrix, diagonal=-block_size).transpose(-2, -1)
        hamiltonian_matrix_hermitian = upper_triangular_matrix - upper_triangular_matrix.transpose(-2, -1)
        hamiltonian_matrix_unflattened = torch.unflatten(hamiltonian_matrix_hermitian, dim=0, sizes=on_site_params.shape[:-3]) # was -2
        return hamiltonian_matrix_unflattened
    