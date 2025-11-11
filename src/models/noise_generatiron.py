from src.hamiltonian.hamiltonian import RepresentationMapping
from src.models.majorana_representation_generator import MajoranaRepresentationHamiltonianConstructor
from src.hamiltonian.torch_hamiltonian import TorchHamiltonian


import torch
import torch.nn as nn


class NoiseGenerator(nn.Module):
    def __init__(self, min_inter_site_interaction_range: int, max_inter_site_interaction_range: int, representation_mapping: RepresentationMapping = RepresentationMapping.none, site_constant: bool = False):
        super(NoiseGenerator, self).__init__()
        self.majorana_rep_ham_constructor = MajoranaRepresentationHamiltonianConstructor(min_inter_site_interaction_range, max_inter_site_interaction_range)
        self.block_size = 4
        self.representation_mapping = representation_mapping
        self.site_constant = site_constant

    def generate_noise(self, n_samples: int, matrix_dim: int, device: torch.device) -> torch.Tensor:
        random_hamiltonian_tensor = self.majorana_rep_ham_constructor.generate_random_hamiltonian(num_samples=n_samples, seq_size=matrix_dim // self.block_size, site_constant=self.site_constant)
        hamiltonians = [TorchHamiltonian.from_2channel_tensor(h).get_hamiltonian_tensor(self.representation_mapping) for h in random_hamiltonian_tensor]
        return torch.stack(hamiltonians).to(device)