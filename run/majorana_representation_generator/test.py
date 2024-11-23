import os
import torch

from src.hamiltonian.hamiltonian import RepresentationMapping
from src.models.majorana_representation_encoder import MajoranaRepresentationHamiltonianExtractor
from src.models.majorana_representation_generator import BaselineHiddenRepresentationGenerator, MajoranaRepresentationHamiltonianGenerator, MajoranaRepresentationHamiltonianConstructor
from src.plots import plot_matrix
from src.torch_utils import TorchHamiltonian


test_dir = './test_majorana_representation_generator'
os.makedirs(test_dir, exist_ok=True)

simple_hidden_rep_generator = BaselineHiddenRepresentationGenerator(**BaselineHiddenRepresentationGenerator.baseline_default_hidden_mlp_config)
majorana_rep_generator = MajoranaRepresentationHamiltonianGenerator(simple_hidden_rep_generator, min_inter_site_interaction_range=2, max_inter_site_interaction_range=3).eval()

input_tensor = torch.randn(1, 100)
generated_hamiltonian = majorana_rep_generator(input_tensor)[0]
H = TorchHamiltonian.from_2channel_tensor(generated_hamiltonian)

plot_matrix(H.get_hamiltonian().real, os.path.join(test_dir, f"random_nn_hamiltonian_real.png"))
plot_matrix(H.get_hamiltonian().imag, os.path.join(test_dir, f"random_nn_hamiltonian_imag.png"))

plot_matrix(H.get_hamiltonian(representation_mapping=RepresentationMapping.inverse_majorana_plus_minus_up_down).real, os.path.join(test_dir, f"random_nn_hamiltonian_real_org_rep.png"))
plot_matrix(H.get_hamiltonian(representation_mapping=RepresentationMapping.inverse_majorana_plus_minus_up_down).imag, os.path.join(test_dir, f"random_nn_hamiltonian_imag_org_rep.png"))

majorana_rep_constructor = MajoranaRepresentationHamiltonianConstructor(min_inter_site_interaction_range=1, max_inter_site_interaction_range=3).eval()
generated_hamiltonian = majorana_rep_constructor.generate_random_hamiltonian(num_samples=1, seq_size=14)[0]
H = TorchHamiltonian.from_2channel_tensor(generated_hamiltonian)

plot_matrix(H.get_hamiltonian().real, os.path.join(test_dir, f"random_hamiltonian_real.png"))
plot_matrix(H.get_hamiltonian().imag, os.path.join(test_dir, f"random_hamiltonian_imag.png"))

plot_matrix(H.get_hamiltonian(representation_mapping=RepresentationMapping.inverse_majorana_plus_minus_up_down).real, os.path.join(test_dir, f"random_hamiltonian_real_org_rep.png"))
plot_matrix(H.get_hamiltonian(representation_mapping=RepresentationMapping.inverse_majorana_plus_minus_up_down).imag, os.path.join(test_dir, f"random_hamiltonian_imag_org_rep.png"))


# Testing random hamiltonian generation and extraction
on_site_params = torch.randn(1, majorana_rep_constructor.num_on_site_params, 14)
inter_site_interaction_range = majorana_rep_constructor.max_inter_site_interaction_range - majorana_rep_constructor.min_inter_site_interaction_range
inter_site_params = torch.randn(1, inter_site_interaction_range, majorana_rep_constructor.num_inter_site_params, 14)
hamiltonian_tensor = majorana_rep_constructor(on_site_params, inter_site_params)

majorana_rep_extractor = MajoranaRepresentationHamiltonianExtractor(min_inter_site_interaction_range=1, max_inter_site_interaction_range=3).eval()
extracted_on_site_params, extracted_inter_site_params = majorana_rep_extractor(hamiltonian_tensor)
assert torch.allclose(on_site_params, extracted_on_site_params), 'On site params are not equal'
assert torch.allclose(inter_site_params, extracted_inter_site_params), 'Inter site params are not equal'