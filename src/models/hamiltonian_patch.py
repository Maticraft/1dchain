import typing as t

import torch
import torch.nn as nn

from src.hamiltonian.hamiltonian_torch_handlers import HamiltonianConstructor, HamiltonianExtractor, HamiltonianParams


class HamiltonianUnpatch(nn.Module):
    def __init__(
        self,
        in_to_out_ratio: int,
        hamitonian_params: HamiltonianParams,
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        **kwargs
    ):
        super(HamiltonianUnpatch, self).__init__()
        self.block_size = 4
        self.min_inter_site_interaction_range = min_inter_site_interaction_range
        self.max_inter_site_interaction_range = max_inter_site_interaction_range
        self.hamiltonian_constructor = HamiltonianConstructor(
            hamitonian_params,
            min_inter_site_interaction_range,
            max_inter_site_interaction_range,
            **kwargs
        )
        self.num_inter_site_params = self.hamiltonian_constructor.num_inter_site_params
        self.num_on_site_params = self.hamiltonian_constructor.num_on_site_params
        self.interaction_range = max_inter_site_interaction_range - min_inter_site_interaction_range
        self.num_total_params = self.interaction_range * self.num_inter_site_params + self.num_on_site_params

        in_size = int(in_to_out_ratio * self.num_total_params)
        self.on_site_converter = nn.Conv1d(in_size, 1, 1)
        self.inter_site_converters = nn.ModuleList([
            nn.Conv1d(in_size, 1, kernel_size=1, stride=1)
            for _ in range(self.min_inter_site_interaction_range, self.max_inter_site_interaction_range)
        ])
    
    def extract_params(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          x: torch.Tensor with shape torch.Tensor with shape (..., (inter_site_interaction_range * num_inter_site_params + num_on_site_params) * seq_size, in_size) 
        Returns:
            Tuple:
            :on_site_params: torch.Tensor with shape (..., num_on_site_params, seq_size)
            :inter_site_params: torch.Tensor with shape (..., inter_site_interaction_range, num_inter_site_params, seq_size)

        '''
        x = x.transpose(-1, -2)
        seq_size = x.shape[-1] // self.num_total_params
        x_on_site = self.on_site_converter(x[..., :seq_size * self.num_on_site_params]).squeeze(-2)
        on_site_params = torch.unflatten(x_on_site, dim=-1, sizes=(self.num_on_site_params, seq_size))

        x_inter_site = x[..., seq_size * self.num_on_site_params:]
        x_inter_site = torch.stack([
            converter(x_inter_site[..., i*seq_size*self.num_inter_site_params:(i+1)*seq_size*self.num_inter_site_params]) for i, converter in enumerate(self.inter_site_converters)
        ], dim=-3).squeeze(-2) # (..., inter_site_interaction_range, num_inter_site_params * seq_size)
        inter_site_params = torch.unflatten(x_inter_site, dim=-1, sizes=(self.num_inter_site_params, seq_size))
        return on_site_params, inter_site_params

    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          x: torch.Tensor with shape torch.Tensor with shape (..., (inter_site_interaction_range * num_inter_site_params + num_on_site_params) * seq_size, in_size) 
        Returns:
          hamiltonian matrix: torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size)
        '''
        on_site_params, inter_site_params = self.extract_params(x)
        hamiltonian = self.hamiltonian_constructor(on_site_params, inter_site_params)
        return hamiltonian
    
    def get_hamiltonian_params(self, x: torch.Tensor) -> t.Dict[str, torch.Tensor]:
        ''''
        Args:
          x: torch.Tensor with shape torch.Tensor with shape (..., (inter_site_interaction_range * num_inter_site_params + num_on_site_params) * seq_size, in_size)
        Returns:
            params_map: Dict[str, torch.Tensor]
        '''
        on_site_params, inter_site_params = self.extract_params(x)
        return self.hamiltonian_constructor.get_params_map(on_site_params, inter_site_params)


class HamiltonianPatchEmbed(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        hamiltonian_params: HamiltonianParams,
        min_inter_site_interaction_range: int = 1,
        max_inter_site_interaction_range: int = 2,
        **kwargs
    ):
        super(HamiltonianPatchEmbed, self).__init__()
        self.min_inter_site_interaction_range = min_inter_site_interaction_range
        self.max_inter_site_interaction_range = max_inter_site_interaction_range
        self.hamiltonian_extractor = HamiltonianExtractor(hamiltonian_params, min_inter_site_interaction_range, max_inter_site_interaction_range, **kwargs)
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

    def patch_params(self, on_site_params: torch.Tensor, inter_site_params: torch.Tensor) -> torch.Tensor:
        '''
        Args:
          on_site_params: torch.Tensor
          inter_site_params: List[torch.Tensor]
        Returns:
          latent_vector: torch.Tensor with shape (..., (inter_site_interaction_range * num_inter_site_params + num_on_site_params) * seq_size, in_size)
        '''
        on_site_params_params_flattened = on_site_params.flatten(start_dim=-2, end_dim=-1).unsqueeze(-2) # (..., 1, num_on_site_params * seq_size)
        on_site_latent = self.on_site_converter(on_site_params_params_flattened) # (..., hidden_size, num_on_site_params * seq_size)
        
        on_site_parameter_classes = torch.arange(self.num_on_site_params, device=on_site_params.device).view(1, -1, 1).expand(-1, -1, on_site_params.shape[-1])
        on_site_parameter_classes_flattened = on_site_parameter_classes.flatten(start_dim=-2, end_dim=-1) # (1, num_on_site_params * seq_size, 1)
        on_site_embedding = self.on_site_embedding_table(on_site_parameter_classes_flattened).transpose(-1, -2) # (1, hidden_size, num_on_site_params * seq_size)
        on_site_latent = on_site_latent + on_site_embedding # (..., hidden_size, num_on_site_params * seq_size)

        inter_site_params_flattened = inter_site_params.flatten(start_dim=-2, end_dim=-1).unsqueeze(-2) # (..., inter_site_interaction_range, 1, num_inter_site_params * seq_size)
        inter_site_latent = torch.cat([
            self.inter_site_converters[i](inter_site_params_flattened[..., i, :, :]) for i in range(inter_site_params.shape[-3])
        ], dim=-1) # (..., hidden_size, inter_site_interaction_range * num_inter_site_params * seq_size)
        
        inter_site_parameter_classes = torch.arange(self.num_inter_site_params, device=inter_site_params.device).view(1, -1, 1).expand(self.interaction_range, -1, inter_site_params.shape[-1])
        inter_site_parameter_classes_flattened = inter_site_parameter_classes.flatten(start_dim=-2, end_dim=-1) # (inter_site_interaction_range, num_inter_site_params * seq_size, 1)
        inter_site_parameter_classes_cat = torch.cat(torch.unbind(inter_site_parameter_classes_flattened, dim=0), dim=0) # (inter_site_interaction_range * num_inter_site_params * seq_size, 1)
        inter_site_embedding = self.inter_site_embedding_table(inter_site_parameter_classes_cat).transpose(-1, -2) # (hidden_size, num_inter_site_params * seq_size)
        inter_site_latent = inter_site_latent + inter_site_embedding # (..., hidden_size, inter_site_interaction_range * num_inter_site_params * seq_size)

        latent_vector = torch.cat([on_site_latent, inter_site_latent], dim=-1) # (..., hidden_size, (inter_site_interaction_range * num_inter_site_params + num_on_site_params) * seq_size)
        return latent_vector.transpose(-1, -2)
          
    def forward(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, t.List[torch.Tensor]]:
        '''
        Args:
          x: torch.Tensor (hamiltonian) with shape torch.Tensor with shape (..., 2, 4 x seq_size, 4 x seq_size) 
        Returns:
          :latent_vector: torch.Tensor with shape (..., num_features)
        '''
        on_site_params, inter_site_params = self.hamiltonian_extractor(x) # (..., num_on_site_params, seq_size), (..., inter_site_interaction_range, num_inter_site_params, seq_size)
        return self.patch_params(on_site_params, inter_site_params)
    
    def forward_from_hamiltonian_params(self, params_map: t.Dict[str, torch.Tensor]) -> torch.Tensor:
        '''
        Args:
          params_map: Dict[str, torch.Tensor]
        Returns:
          :latent_vector: torch.Tensor with shape (..., num_features)
        '''
        on_site_params, inter_site_params = self.hamiltonian_extractor.from_params_map(params_map)
        return self.patch_params(on_site_params, inter_site_params)