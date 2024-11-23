from abc import abstractmethod
import json_fix
import typing as t
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.hamiltonian.hamiltonian_torch_handlers import ALL_PAIRS
from src.hamiltonian.hamiltonian_torch_handlers import BlockExtractor, get_strip
from src.models.base_models import MLP, UNetLikeEncoder

class HiddenRepresentationEncoder(nn.Module):
    def __init__(self, hidden_representation_size: int):
        super(HiddenRepresentationEncoder, self).__init__()
        self.hidden_representation_size = hidden_representation_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x: torch.Tensor with shape (..., hidden_representation_size, seq_size)
        Returns:
          latent_vector: torch.Tensor with shape (..., num_features) or tuples of tensors if skip connections are used
        '''
        raise NotImplementedError('HiddenRepresentationGenerator.forward not implemented')


class BaselineHiddenRepresentationEncoder(HiddenRepresentationEncoder):
    def __init__(self, output_size: int, hidden_representation_size: int, **hiden_mlp_config: t.Dict[str, t.Any]):
        super(BaselineHiddenRepresentationEncoder, self).__init__(hidden_representation_size)
        self.hidden_mlp_config = hiden_mlp_config
        mlp_config = {
            'layers_num': hiden_mlp_config.get('layers_num', 3),
            'input_size': hidden_representation_size * hiden_mlp_config.get('seq_size', 14),
            'hidden_size': hiden_mlp_config.get('hidden_size', 64),
            'output_size': output_size,
            'activation': hiden_mlp_config.get('activation', 'relu'),
            'final_activation': hiden_mlp_config.get('final_activation', 'none')
        }
        self.mlp = MLP(**mlp_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x: torch.Tensor with shape (..., hidden_size, seq_size)
        Returns:
          :latent_vector: torch.Tensor with shape (..., num_features)
        '''
        x = x.view(*x.shape[:-2], -1)
        return self.mlp(x)
    
    # make model json serializable
    def __json__(self):
        return {
            'hidden_representation_size': self.hidden_representation_size,
            'hidden_mlp_config': self.hidden_mlp_config
        }
    

class SiteIndependentHiddenRepresentationEncoder(HiddenRepresentationEncoder):
    def __init__(self, output_size: int, hidden_representation_size: int, **hiden_mlp_config: t.Dict[str, t.Any]):
        super(SiteIndependentHiddenRepresentationEncoder, self).__init__(hidden_representation_size)
        self.hidden_mlp_config = hiden_mlp_config
        self.seq_size = hiden_mlp_config.get('seq_size', 14)
        mlp_config = {
            'layers_num': hiden_mlp_config.get('layers_num', 3),
            'input_size': hidden_representation_size,
            'hidden_size': hiden_mlp_config.get('hidden_size', 64),
            'output_size': output_size,
            'activation': hiden_mlp_config.get('activation', 'relu'),
            'final_activation': hiden_mlp_config.get('final_activation', 'none')
        }
        self.mlps = nn.ModuleList([MLP(**mlp_config) for _ in range(self.seq_size)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x: torch.Tensor with shape (..., hidden_size, seq_size)
        Returns:
          :latent_vector: torch.Tensor with shape (..., num_features, seq_size)
        '''
        encoded_sites = torch.stack([mlp(x[..., i]) for i, mlp in enumerate(self.mlps)], dim=-1)
        return encoded_sites
    
    # make model json serializable
    def __json__(self):
        return {
            'hidden_representation_size': self.hidden_representation_size,
            'hidden_mlp_config': self.hidden_mlp_config
        }
    

class UNetLikeSiteIndependentHiddenRepresentationEncoder(HiddenRepresentationEncoder):
    def __init__(self, output_size: int, hidden_representation_size: int, unet_layers: int, unet_hidden_size: int, seq_size: int, repeat_along_seq: bool = False):
        super(UNetLikeSiteIndependentHiddenRepresentationEncoder, self).__init__(hidden_representation_size)
        self.hidden_representation_size = hidden_representation_size
        self.seq_size = seq_size
        self.repeat_along_seq = repeat_along_seq
        self.unet_encoder_config = {
            'layers_num': unet_layers,
            'input_size': hidden_representation_size,
            'hidden_size': unet_hidden_size,
            'output_size': output_size,
        }
        if not self.repeat_along_seq:
            self.unet_encoders = nn.ModuleList([UNetLikeEncoder(**self.unet_encoder_config) for _ in range(self.seq_size)])
        else:
            self.unet_encoder = UNetLikeEncoder(**self.unet_encoder_config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          x: torch.Tensor with shape (..., hidden_size, seq_size)
        Returns:
          :latent_vector: torch.Tensor with shape (..., num_features, seq_size)
        '''
        encoded_sites = []
        for i in range(self.seq_size):
            if self.repeat_along_seq:
                encoded_site = self.unet_encoder(x[..., i])
            else:
                encoded_site = self.unet_encoders[i](x[..., i])
            encoded_sites.append(encoded_site)
        
        encoded_sites = [
            torch.stack([encoded_site[i] for encoded_site in encoded_sites], dim=-1)
            for i in range(len(encoded_sites[0]))
        ]
        return encoded_sites
    
    # make model json serializable
    def __json__(self):
        return {
            'hidden_representation_size': self.hidden_representation_size,
            'seq_size': self.seq_size,
            'unet': self.unet_encoder_config
        }



class MajoranaRepresentationHamiltonianEncoder(nn.Module):
    def __init__(
        self,
        hidden_representation_encoder: HiddenRepresentationEncoder,
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        **kwargs: t.Any
    ):
        super(MajoranaRepresentationHamiltonianEncoder, self).__init__()
        # Extractor
        self.majorana_hamiltonian_extractor = MajoranaRepresentationHamiltonianExtractor(min_inter_site_interaction_range, max_inter_site_interaction_range)
        self.num_on_site_params = self.majorana_hamiltonian_extractor.num_on_site_params
        self.num_inter_site_params = self.majorana_hamiltonian_extractor.num_inter_site_params
        self.min_inter_site_interaction_range = min_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.
        self.max_inter_site_interaction_range = max_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.

        # NNs
        self.interaction_range = max_inter_site_interaction_range - min_inter_site_interaction_range
        self.num_independent_strips = self.interaction_range + 1 # on site + inter site
        self.hidden_representation_encoder: HiddenRepresentationEncoder = hidden_representation_encoder
        self.hidden_channels = self.hidden_representation_encoder.hidden_representation_size
        self.on_site_converter = nn.Conv1d(self.num_on_site_params, self.hidden_channels // self.num_independent_strips, 1, bias=False)
        self.inter_site_converters = nn.ModuleList([
            nn.Conv1d(self.num_inter_site_params, self.hidden_channels // self.num_independent_strips, kernel_size=1, stride=1, dilation=interaction_range, bias=False)
            for interaction_range in range(self.min_inter_site_interaction_range, self.max_inter_site_interaction_range)
        ])

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          x: torch.Tensor (hamiltonian) with shape torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size) 
        Returns:
          :latent_vector: torch.Tensor with shape (..., num_features)
        '''
        on_site_params, inter_site_params =  self.majorana_hamiltonian_extractor(x)
        latent_vector = self._nn_forward(on_site_params, inter_site_params)
        return latent_vector

    def _nn_forward(self, on_site_params: torch.Tensor, inter_site_params: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        Returns:
          :latent_vector: torch.Tensor with shape (..., num_features)
        '''
        on_site_latent = self.on_site_converter(on_site_params).unsqueeze(-3) # (..., 1, hidden_size / num_independent_strips, seq_size)
        inter_site_latent = torch.stack([
            self.inter_site_converters[i](inter_site_params[..., i, :, :]) for i in range(inter_site_params.shape[-3])
        ], dim=-3) # (..., inter_site_interaction_range, hidden_size / num_independent_strips, seq_size)
        latent_vector = torch.cat([on_site_latent, inter_site_latent], dim=-3).flatten(start_dim=-3, end_dim=-2) # (..., hidden_size, seq_size)
        latent_vector = self.hidden_representation_encoder(latent_vector)
        return latent_vector

  
class MajoranaRepresentationHamiltonianExtractor(nn.Module):
    def __init__(
        self,
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        **kwargs: t.Any
    ):
        super(MajoranaRepresentationHamiltonianExtractor, self).__init__()
        self.on_site_block_names = ['xiy', 'iy1', 'iyz', 'ziy']
        self.num_on_site_params = len(self.on_site_block_names)
        self.inter_site_block_names = ['1x', '1iy', '1z', 'iy1', 'iyx']
        self.num_inter_site_params = len(self.inter_site_block_names)
        self.min_inter_site_interaction_range = min_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.
        self.max_inter_site_interaction_range = max_inter_site_interaction_range # 1 for on site only, 2 for nearest neighbors, 3 for next nearest neighbors, etc.


    def forward(self, hamiltonian: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
          hamiltonian matrix: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)     
        Returns:
          :on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          :inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        '''
        on_site_params, inter_site_params = self._extract_params_from_hamiltonian_matrix(hamiltonian)
        return on_site_params, inter_site_params
    
    
    def _extract_params_from_hamiltonian_matrix(self, hamiltonian: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor]:
        '''
        Args:
          hamiltonian matrix: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)     
        Returns:
          :on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
          :inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        '''
        block_size = 4
        on_site_strip = get_strip(hamiltonian, 0, 'hamiltonian', block_size)
        on_site_params = BlockExtractor.extract_block_sequences(on_site_strip[:, 1], self.on_site_block_names)

        inter_site_strips = torch.stack([
            get_strip(hamiltonian, strip_idx, 'hamiltonian', block_size)
            for strip_idx in range(self.min_inter_site_interaction_range, self.max_inter_site_interaction_range)
        ], dim=1)
        inter_site_params = BlockExtractor.extract_block_sequences(inter_site_strips.flatten(0, 1)[:, 1], self.inter_site_block_names)
        inter_site_params = torch.unflatten(inter_site_params, dim=0, sizes=(inter_site_strips.shape[0], inter_site_strips.shape[1]))
        return on_site_params, inter_site_params
    

class MajoranaRepresentationPatchEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        **kwargs: t.Any
    ):
        super(MajoranaRepresentationPatchEmbed, self).__init__()
        self.min_inter_site_interaction_range = min_inter_site_interaction_range
        self.max_inter_site_interaction_range = max_inter_site_interaction_range
        self.hamiltonian_extractor = MajoranaRepresentationHamiltonianExtractor(min_inter_site_interaction_range, max_inter_site_interaction_range)
        self.num_inter_site_params = self.hamiltonian_extractor.num_inter_site_params
        self.num_on_site_params = self.hamiltonian_extractor.num_on_site_params
        self.interaction_range = max_inter_site_interaction_range - min_inter_site_interaction_range
        self.num_independent_strips = self.interaction_range + 1 # on site + inter site

        self.on_site_embedding_table = nn.Embedding(self.num_on_site_params, hidden_size)
        self.inter_site_embedding_table = nn.Embedding(self.num_inter_site_params, hidden_size)

        self.on_site_converter = nn.Conv1d(1, hidden_size, 1, bias=False)
        self.inter_site_converters = nn.ModuleList([
            nn.Conv1d(1, hidden_size, kernel_size=1, stride=1, bias=False)
            for _ in range(self.min_inter_site_interaction_range, self.max_inter_site_interaction_range)
        ])

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          x: torch.Tensor (hamiltonian) with shape torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size) 
        Returns:
          :latent_vector: torch.Tensor with shape (..., num_features)
        '''
        on_site_params, inter_site_params = self.hamiltonian_extractor(x) # (..., num_on_site_params, seq_size), (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        on_site_params_params_flattened = on_site_params.flatten(start_dim=-2, end_dim=-1).unsqueeze(-2) # (..., 1, num_on_site_params * seq_size)
        on_site_latent = self.on_site_converter(on_site_params_params_flattened) # (..., hidden_size, num_on_site_params * seq_size)
        
        on_site_parameter_classes = torch.arange(self.num_on_site_params, device=x.device).view(1, -1, 1).expand(-1, -1, on_site_params.shape[-1])
        on_site_parameter_classes_flattened = on_site_parameter_classes.flatten(start_dim=-2, end_dim=-1) # (1, num_on_site_params * seq_size, 1)
        on_site_embedding = self.on_site_embedding_table(on_site_parameter_classes_flattened).transpose(-1, -2) # (1, hidden_size, num_on_site_params * seq_size)
        on_site_latent = on_site_latent + on_site_embedding # (..., hidden_size, num_on_site_params * seq_size)

        inter_site_params_flattened = inter_site_params.flatten(start_dim=-2, end_dim=-1).unsqueeze(-2) # (..., inter_site_interaction_range, 1, num_inter_site_params * seq_size)
        inter_site_latent = torch.cat([
            self.inter_site_converters[i](inter_site_params_flattened[..., i, :, :]) for i in range(inter_site_params.shape[-3])
        ], dim=-1) # (..., hidden_size, inter_site_interaction_range * num_inter_site_params * seq_size)
        
        inter_site_parameter_classes = torch.arange(self.num_inter_site_params, device=x.device).view(1, -1, 1).expand(self.interaction_range, -1, inter_site_params.shape[-1])
        inter_site_parameter_classes_flattened = inter_site_parameter_classes.flatten(start_dim=-2, end_dim=-1) # (inter_site_interaction_range, num_inter_site_params * seq_size, 1)
        inter_site_parameter_classes_cat = torch.cat(torch.unbind(inter_site_parameter_classes_flattened, dim=0), dim=0) # (inter_site_interaction_range * num_inter_site_params * seq_size, 1)
        inter_site_embedding = self.inter_site_embedding_table(inter_site_parameter_classes_cat).transpose(-1, -2) # (hidden_size, num_inter_site_params * seq_size)
        inter_site_latent = inter_site_latent + inter_site_embedding # (..., hidden_size, inter_site_interaction_range * num_inter_site_params * seq_size)

        latent_vector = torch.cat([on_site_latent, inter_site_latent], dim=-1) # (..., hidden_size, (inter_site_interaction_range * num_inter_site_params + num_on_site_params) * seq_size)
        return latent_vector.transpose(-1, -2)
