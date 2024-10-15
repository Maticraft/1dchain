import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.hamiltonian.hamiltonian import RepresentationMapping
from src.models.majorana_representation_generator import MajoranaRepresentationHamiltonianConstructor
from src.torch_utils import TorchHamiltonian


class Embedding(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super(Embedding, self).__init__()
        self.embedding = nn.Linear(n_in, n_out, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
    

class NoiseGenerator(nn.Module):
    def __init__(self, min_inter_site_interaction_range: int, max_inter_site_interaction_range: int, representation_mapping: RepresentationMapping = RepresentationMapping.none):
        super(NoiseGenerator, self).__init__()
        self.majorana_rep_ham_constructor = MajoranaRepresentationHamiltonianConstructor(min_inter_site_interaction_range, max_inter_site_interaction_range)
        self.block_size = 4
        self.representation_mapping = representation_mapping

    def generate_noise(self, n_samples: int, matrix_dim: int, device: torch.device) -> torch.Tensor:
        random_hamiltonian_tensor = self.majorana_rep_ham_constructor.generate_random_hamiltonian(num_samples=n_samples, seq_size=matrix_dim // self.block_size)
        hamiltonians = [TorchHamiltonian.from_2channel_tensor(h).get_hamiltonian_tensor(self.representation_mapping) for h in random_hamiltonian_tensor]
        return torch.stack(hamiltonians).to(device)


class DiffusionAutoencoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, latent_shapes: t.Tuple[int, ...] = (100,), n_context_feat: t.Optional[int] = None):
        super(DiffusionAutoencoder, self).__init__()
        self.latent_shapes = latent_shapes
        self.n_context_feat = n_context_feat

        self.time_embeddings = nn.ModuleList([Embedding(1, latent_shape) for latent_shape in self.latent_shapes])
        if self.n_context_feat is not None:
            self.context_embeddings = nn.ModuleList([Embedding(self.n_context_feat, latent_shape) for latent_shape in self.latent_shapes])
        else:
            self.context_embeddings = [lambda x: torch.ones(1, latent_shape) for latent_shape in self.latent_shapes]
        self.encoder = encoder # if UNet like model, then encoder should return a tuple of tensors representing outputs for subsequent skip layers
        self.decoder = decoder # if UNet like model, then decoder should accept a tuple of tensors representing inputs for subsequent skip layers


    def forward(self, x: torch.Tensor, time_step: torch.Tensor, context_vector: t.Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x : (batch, n_feat, h, w) : input image
        time_step : (batch, 1)      : time step
        context_vector : (batch, n_cfeat)    : context label
        """
        encoded = self.encoder(x)
        if len(self.latent_shapes) == 1:
            encoded = [encoded]

        encoded_with_embeddings = [
            unsqueeze_like(context_embedding(context_vector).to(encoded_i.device), encoded_i) * encoded_i + unsqueeze_like(time_embedding(time_step), encoded_i)
            for encoded_i, time_embedding, context_embedding in zip(encoded, self.time_embeddings, self.context_embeddings)
        ]

        if len(self.latent_shapes) == 1:
            encoded_with_embeddings = encoded_with_embeddings[0]

        decoded = self.decoder(encoded_with_embeddings)
        return decoded
    

def unsqueeze_like(x: torch.Tensor, target: torch.Tensor, start_dim: int = 2):
    unsqueezed_shape = [1]*(len(target.shape) - start_dim)
    return x.view(*x.shape[:start_dim], *unsqueezed_shape)


def train_diffusion_model(
    model: DiffusionAutoencoder,
    noise_generator: NoiseGenerator,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    timesteps: int = 500,
    beta1: float = 1e-4,
    beta2: float = 1e-2,

) -> nn.Module:

    model.to(device)
    model.train()

    total_loss = 0.

    # construct DDPM noise schedule
    _, _, ab_t = construct_ddpm_noise_schedule(timesteps, beta1, beta2, device)

    print(f'Epoch: {epoch}')
    for (x, y), _ in tqdm(train_loader, 'Training diffusion model'):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        
        # perturb data
        noise = noise_generator.generate_noise(x.shape[0], x.shape[-1], device)
        t = torch.randint(1, timesteps + 1, (x.shape[0],)).to(device) 
        x_perturbed = ab_t.sqrt()[t, None, None, None] * x + (1 - ab_t[t, None, None, None]) * noise
        
        # use network to recover noise
        pred_noise = model(x_perturbed, t.unsqueeze(-1) / timesteps, context_vector=y)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_noise, noise)
        total_loss += loss.item()
        loss.backward()
        
        optimizer.step()

    total_loss /= len(train_loader)
    print(f'Loss: {total_loss}')
    print()
    return total_loss


def construct_ddpm_noise_schedule(
    timesteps: int = 500,
    beta1: float = 1e-4,
    beta2: float = 1e-2,
    device: torch.device = torch.device('cpu')
) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    
    b_t = (beta2 - beta1) * torch.linspace(0, 1, timesteps + 1, device=device) + beta1
    a_t = 1 - b_t
    ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
    ab_t[0] = 1
    return b_t, a_t, ab_t


@torch.no_grad()
def sample_ddpm(
    model: DiffusionAutoencoder,
    noise_generator: NoiseGenerator,
    n_samples: int,
    matrix_dim: int,
    timesteps: int = 500,
    beta1: float = 1e-4,
    beta2: float = 1e-2,
    device: torch.device = torch.device('cpu'),
    context_vector: t.Optional[torch.Tensor] = None,
):
    model.to(device)
    b_t, a_t, ab_t = construct_ddpm_noise_schedule(timesteps, beta1, beta2, device)

    # x_T ~ N(0, 1), sample initial noise
    samples = noise_generator.generate_noise(n_samples, matrix_dim, device)
    context_vector = context_vector.to(device).unsqueeze(0).expand(n_samples, -1) if context_vector is not None else None

    for i in range(timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None].to(device)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = noise_generator.generate_noise(n_samples, matrix_dim, device) if i > 1 else 0

        pred_noise = model(samples, t, context_vector)    # predict noise e_(x_t,t)
        
        noise = b_t.sqrt()[i] * z
        mean = (samples - pred_noise * ((1 - a_t[i]) / (1 - ab_t[i]).sqrt())) / a_t[i].sqrt()
        samples = mean + noise

    return samples


@torch.no_grad()
def sample_ddim(
    model: DiffusionAutoencoder,
    noise_generator: NoiseGenerator,
    n_samples: int,
    matrix_dim: int,
    timesteps: int = 500,
    implicit_steps: int = 20,
    beta1: float = 1e-4,
    beta2: float = 1e-2,
    device: torch.device = torch.device('cpu'),
):
    model.to(device)
    _, _, ab_t = construct_ddpm_noise_schedule(timesteps, beta1, beta2, device)

    # x_T ~ N(0, 1), sample initial noise
    samples = noise_generator.generate_noise(n_samples, matrix_dim, device)

    step_size = timesteps // implicit_steps
    for i in range(timesteps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / timesteps])[:, None, None, None].to(device)

        pred_noise = model(samples, t.unsqueeze(-1))    # predict noise e_(x_t,t)

        # denoise ddim
        ab = ab_t[i]
        ab_prev = ab_t[i - step_size]
        
        x0_pred = ab_prev.sqrt() / ab.sqrt() * (samples - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        samples = x0_pred + dir_xt

    return samples
