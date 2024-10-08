import os
import torch

from src.hamiltonian.hamiltonian import RepresentationMapping
from src.models.majorana_representation_generator import BaselineHiddenRepresentationGenerator, MajoranaRepresentationHamiltonianGenerator
from src.plots import plot_matrix
from src.torch_utils import TorchHamiltonian


test_dir = './test_majorana_representation_generator'
os.makedirs(test_dir, exist_ok=True)

simple_hidden_rep_generator = BaselineHiddenRepresentationGenerator(**BaselineHiddenRepresentationGenerator.baseline_default_hidden_mlp_config)
majorana_rep_generator = MajoranaRepresentationHamiltonianGenerator(simple_hidden_rep_generator, min_inter_site_interaction_range=2, max_inter_site_interaction_range=3).eval()

input_tensor = torch.randn(1, 100)
generated_hamiltonian = majorana_rep_generator(input_tensor)[0]
H = TorchHamiltonian.from_2channel_tensor(generated_hamiltonian)

plot_matrix(H.get_hamiltonian().real, os.path.join(test_dir, f"random_hamiltonian_real.png"))
plot_matrix(H.get_hamiltonian().imag, os.path.join(test_dir, f"random_hamiltonian_imag.png"))

plot_matrix(H.get_hamiltonian(representation_mapping=RepresentationMapping.inverse_majorana_plus_minus_up_down).real, os.path.join(test_dir, f"random_hamiltonian_real_org_rep.png"))
plot_matrix(H.get_hamiltonian(representation_mapping=RepresentationMapping.inverse_majorana_plus_minus_up_down).imag, os.path.join(test_dir, f"random_hamiltonian_imag_org_rep.png"))