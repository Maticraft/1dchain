from abc import abstractmethod
import jsonpickle
import json_fix
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hamiltonian.hamiltonian_torch_handlers import ALL_PAIRS
from src.hamiltonian.hamiltonian_torch_handlers import BlockConstructor, get_matrix_from_strips
from src.models.base_models import MLP, UNetLikeDecoder

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
        'seq_size': 14,
        'activation': 'relu',
        'final_activation': 'none'
    }
    def __init__(self, hidden_representation_size: int = 64, **hiden_mlp_config: t.Dict[str, t.Any]):
        super(BaselineHiddenRepresentationGenerator, self).__init__(hidden_representation_size)
        self.hidden_mlp_config = hiden_mlp_config
        mlp_config = {
            'layers_num': hiden_mlp_config.get('layers_num', 3),
            'input_size': hiden_mlp_config.get('input_size', 100),
            'hidden_size': hiden_mlp_config.get('hidden_size', 64),
            'output_size': hidden_representation_size, #* hiden_mlp_config.get('seq_size', 14),
            'activation': hiden_mlp_config.get('activation', 'relu'),
            'final_activation': hiden_mlp_config.get('final_activation', 'none')
        }
        self.mlp = MLP(**mlp_config)
        self.seq_size = hiden_mlp_config.get('seq_size', 14)
        # self.mlps = nn.ModuleList([MLP(**hiden_mlp_config) for _ in range(num_hidden_mlps)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x: torch.Tensor with shape (..., num_features)
        Returns:
          torch.Tensor with shape (..., hidden_representation_size, seq_size)
        '''
        hidden_representation = self.mlp(x)
        return hidden_representation.view(*hidden_representation.shape[:-1], self.hidden_representation_size, 1).expand(*hidden_representation.shape[:-1], -1, self.seq_size)
    
    # make model json serializable
    def __json__(self):
        return {
            'num_hidden_mlps': self.hidden_representation_size,
            'hidden_mlp_config': self.hidden_mlp_config
        }
    

class SiteIndependentRepresentationGenerator(HiddenRepresentationGenerator):
    baseline_default_hidden_mlp_config = {
        'layers_num': 3,
        'input_size': 100,
        'hidden_size': 64,
        'seq_size': 14,
        'activation': 'relu',
        'final_activation': 'none'
      }
    def __init__(self, hidden_representation_size: int = 64, **hiden_mlp_config: t.Dict[str, t.Any]):
        super(SiteIndependentRepresentationGenerator, self).__init__(hidden_representation_size)
        self.hidden_mlp_config = hiden_mlp_config
        mlp_config = {
            'layers_num': hiden_mlp_config.get('layers_num', 3),
            'input_size': hiden_mlp_config.get('input_size', 100),
            'hidden_size': hiden_mlp_config.get('hidden_size', 64),
            'output_size': hidden_representation_size, #* hiden_mlp_config.get('seq_size', 14),
            'activation': hiden_mlp_config.get('activation', 'relu'),
            'final_activation': hiden_mlp_config.get('final_activation', 'none')
        }
        self.seq_size = hiden_mlp_config.get('seq_size', 14)
        self.mlps = nn.ModuleList([MLP(**mlp_config) for _ in range(self.seq_size)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x: torch.Tensor with shape (..., num_features, seq_size)
        Returns:
          torch.Tensor with shape (..., hidden_representation_size, seq_size)
        '''
        hidden_representation = torch.stack([mlp(x[..., i]) for i, mlp in enumerate(self.mlps)], dim=-1)
        return hidden_representation
    
    # make model json serializable
    def __json__(self):
        return {
            'num_hidden_mlps': self.hidden_representation_size,
            'hidden_mlp_config': self.hidden_mlp_config
        }
    

class UNetLikeSiteIndependentHiddenRepresentationGenerator(HiddenRepresentationGenerator):
    def __init__(self, input_size: int, hidden_representation_size: int, unet_layers: int, unet_hidden_size: int, seq_size: int, repeat_along_seq: bool = False):
        super(UNetLikeSiteIndependentHiddenRepresentationGenerator, self).__init__(hidden_representation_size)
        self.hidden_representation_size = hidden_representation_size
        self.seq_size = seq_size
        self.repeat_along_seq = repeat_along_seq
        self.unet_decoder_config = {
            'layers_num': unet_layers,
            'input_size': input_size,
            'hidden_size': unet_hidden_size,
            'output_size': hidden_representation_size,
        }
        if self.repeat_along_seq:
            self.unet_decoder = UNetLikeDecoder(**self.unet_decoder_config)
        else:
            self.unet_decoders = nn.ModuleList([UNetLikeDecoder(**self.unet_decoder_config) for _ in range(self.seq_size)])

    def forward(self, x: t.List[torch.Tensor]) -> torch.Tensor:
        '''
        Args:
          x: List[torch.Tensor] with shape (..., hidden_size, seq_size)
        Returns:
          :latent_vector: torch.Tensor with shape (..., num_features, seq_size)
        '''
        decoded_sites = []
        for i in range(self.seq_size):
            decoder_input = [x_l[..., i] for x_l in x]
            if self.repeat_along_seq:
                decoded_site = self.unet_decoder(decoder_input)
            else:
                decoded_site = self.unet_decoders[i](decoder_input)
            decoded_sites.append(decoded_site)
        
        return torch.stack(decoded_sites, dim=-1)
    
    # make model json serializable
    def __json__(self):
        return {
            'hidden_representation_size': self.hidden_representation_size,
            'seq_size': self.seq_size,
            'unet': self.unet_decoder_config
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
        # Constructor
        self.majorana_hamiltonian_constructor = MajoranaRepresentationHamiltonianConstructor(min_inter_site_interaction_range, max_inter_site_interaction_range)
        self.num_on_site_params = self.majorana_hamiltonian_constructor.num_on_site_params
        self.num_inter_site_params = self.majorana_hamiltonian_constructor.num_inter_site_params
        self.min_inter_site_interaction_range = min_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.
        self.max_inter_site_interaction_range = max_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.

        # NNs
        self.hidden_representation_generator: HiddenRepresentationGenerator = hidden_representation_generator
        self.hidden_channels = self.hidden_representation_generator.hidden_representation_size
        self.interaction_range = max_inter_site_interaction_range - min_inter_site_interaction_range
        self.num_independent_strips = self.interaction_range + 1 # on site + inter site
        self.num_hidden_channels_per_strip = self.hidden_channels // self.num_independent_strips
        self.on_site_converter = nn.Conv1d(self.num_hidden_channels_per_strip, self.num_on_site_params, 1)
        self.inter_site_converters = nn.ModuleList([
            nn.Conv1d(self.num_hidden_channels_per_strip, self.num_inter_site_params, kernel_size=1, stride=1, dilation=interaction_range)
            for interaction_range in range(self.min_inter_site_interaction_range, self.max_inter_site_interaction_range)
        ])
        # self.inter_site_convs = nn.ConvTranspose2d(self.hidden_channels, 1, 4, stride=4)
        # self.on_site_convs = nn.ConvTranspose2d(self.hidden_channels, 1, 4, stride=4)

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          x: torch.Tensor with shape (..., num_features)
        Returns:
          hamiltonian: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
        '''
        on_site_params, inter_site_params = self._nn_forward(x)
        hamiltonian = self.majorana_hamiltonian_constructor(on_site_params, inter_site_params)
        return hamiltonian

    def _nn_forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
          x: torch.Tensor with shape (..., num_features)
        Returns:
          :on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          :inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        '''
        hidden_representation = self.hidden_representation_generator(x) # (..., hidden_channels, seq_size)
        on_site_params = self.on_site_converter(hidden_representation[..., :self.num_hidden_channels_per_strip, :])
        # on_site_params = self.on_site_convs(hidden_representation.unsqueeze(-2)).squeeze(-3)
        inter_site_params = torch.stack([
            converter(hidden_representation[..., (i+1)*self.num_hidden_channels_per_strip: (i+2)*self.num_hidden_channels_per_strip, :]) for i, converter in enumerate(self.inter_site_converters)
        ], dim=-3) # (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        # inter_site_params = torch.cat([self.inter_site_convs(inter_site_params[..., i, :, :].unsqueeze(-2)) for i in range(inter_site_params.shape[-3])], dim=-3)
        return on_site_params, inter_site_params  

  
class MajoranaRepresentationHamiltonianConstructor(nn.Module):
    def __init__(
        self,
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        **kwargs: t.Any
    ):
        super(MajoranaRepresentationHamiltonianConstructor, self).__init__()
        self.on_site_block_names = ['xiy', 'iy1', 'iyz', 'ziy']
        self.num_on_site_params = len(self.on_site_block_names)
        self.inter_site_block_names = ['1x', '1iy', '1z', 'iy1', 'iyx']
        self.num_inter_site_params = len(self.inter_site_block_names)
        self.min_inter_site_interaction_range = min_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.
        self.max_inter_site_interaction_range = max_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.


    def forward(self, on_site_params: torch.Tensor, inter_site_params: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        Returns:
          :hamiltonian matrix: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
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
          :hamiltonian: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
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
          :hamiltonian matrix: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
        '''
        block_size = 4
        flattened_on_site_params = on_site_params.flatten(start_dim=0, end_dim=-3)
        on_site_block_seq = torch.stack((
            torch.zeros(flattened_on_site_params.shape[0], block_size, block_size*flattened_on_site_params.shape[-1], device=flattened_on_site_params.device), # real part is zeroed in majorana representation
            BlockConstructor.generate_block_sequence(flattened_on_site_params.transpose(0, 1), self.on_site_block_names, lower_block_transpose=False) # imaginary part
        ), dim=-3)

        flattened_inter_site_params = inter_site_params.flatten(start_dim=0, end_dim=-4)
        # flattened_inter_site_params_expanded = flattened_inter_site_params.repeat(1, 1, len(self.inter_site_block_names), 1)
        # flattened_inter_site_params_expanded = flattened_inter_site_params_expanded * torch.tensor(self.inter_site_signs).to(flattened_inter_site_params_expanded.device).unsqueeze(-1)
        inter_site_block_seq = torch.cat([
            torch.stack((
                torch.zeros(flattened_inter_site_params.shape[0], block_size, block_size*flattened_inter_site_params.shape[-1], device=flattened_inter_site_params.device), # real part is zeroed in majorana representation
                BlockConstructor.generate_block_sequence(flattened_inter_site_params[:, inter_site_interaction_range_idx].transpose(0, 1), self.inter_site_block_names) # imaginary part
            ), dim=-3)
            for inter_site_interaction_range_idx in range(flattened_inter_site_params.shape[-3])
        ], dim=-3)

        skipped_inter_site_block_seq = [
            torch.zeros(flattened_inter_site_params.shape[0], 2, block_size, inter_site_block_seq.shape[-1], device=flattened_inter_site_params.device)
            for _ in range(self.min_inter_site_interaction_range - 1)
        ]
        if len(skipped_inter_site_block_seq) > 0:
            inter_site_block_seq = torch.cat([*skipped_inter_site_block_seq, inter_site_block_seq], dim=-3)

        strips = torch.cat([torch.zeros_like(inter_site_block_seq), on_site_block_seq, inter_site_block_seq], dim=-3)
        hamiltonian_matrix = get_matrix_from_strips(strips, on_site_params.shape[-1], block_size)
        upper_triangular_matrix = torch.triu(hamiltonian_matrix) - torch.tril(hamiltonian_matrix, diagonal=-block_size).transpose(-2, -1) # because of periodic inter site elements
        hamiltonian_matrix_hermitian = upper_triangular_matrix - upper_triangular_matrix.transpose(-2, -1)
        hamiltonian_matrix_unflattened = torch.unflatten(hamiltonian_matrix_hermitian, dim=0, sizes=on_site_params.shape[:-2])
        return hamiltonian_matrix_unflattened
    
class MajoranaRepresentationUnpatch(nn.Module):
    def __init__(
        self,
        in_size: int,
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        **kwargs: t.Any
    ):
        super(MajoranaRepresentationUnpatch, self).__init__()
        self.min_inter_site_interaction_range = min_inter_site_interaction_range
        self.max_inter_site_interaction_range = max_inter_site_interaction_range
        self.hamiltonian_constructor = MajoranaRepresentationHamiltonianConstructor(min_inter_site_interaction_range, max_inter_site_interaction_range)
        self.num_inter_site_params = self.hamiltonian_constructor.num_inter_site_params
        self.num_on_site_params = self.hamiltonian_constructor.num_on_site_params
        self.interaction_range = max_inter_site_interaction_range - min_inter_site_interaction_range
        self.seq_size_multiplication_factor = self.interaction_range * self.num_inter_site_params + self.num_on_site_params

        self.on_site_converter = nn.Conv1d(in_size, 1, 1)
        self.inter_site_converters = nn.ModuleList([
            nn.Conv1d(in_size, 1, kernel_size=1, stride=1)
            for _ in range(self.min_inter_site_interaction_range, self.max_inter_site_interaction_range)
        ])

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          x: torch.Tensor with shape torch.Tensor with shape (..., (inter_site_interaction_range * num_inter_site_params + num_on_site_params) * seq_size, in_size) 
        Returns:
          :hamiltonian matrix: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
        '''
        x = x.transpose(-1, -2)
        seq_size = x.shape[-1] // self.seq_size_multiplication_factor
        x_on_site = self.on_site_converter(x[..., :seq_size * self.num_on_site_params]).squeeze(-2)
        on_site_params = torch.unflatten(x_on_site, dim=-1, sizes=(self.num_on_site_params, seq_size))

        x_inter_site = x[..., seq_size * self.num_on_site_params:]
        x_inter_site = torch.stack([
            converter(x_inter_site[..., i*seq_size*self.num_inter_site_params:(i+1)*seq_size*self.num_inter_site_params]) for i, converter in enumerate(self.inter_site_converters)
        ], dim=-3).squeeze(-2) # (..., inter_site_interaction_range, num_inter_site_params * seq_size)
        inter_site_params = torch.unflatten(x_inter_site, dim=-1, sizes=(self.num_inter_site_params, seq_size))
        hamiltonian = self.hamiltonian_constructor(on_site_params, inter_site_params)
        return hamiltonian
