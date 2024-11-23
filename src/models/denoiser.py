from abc import abstractmethod, ABC
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.noise_generatiron import NoiseGenerator
from src.hamiltonian.hamiltonian import RepresentationMapping
from src.models.majorana_representation_generator import MajoranaRepresentationHamiltonianConstructor
from src.torch_utils import TorchHamiltonian

 
def train_denoising_model(
    model: nn.Module,
    noise_generator: NoiseGenerator,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    noise_max_strength: float = 0.1,
    mock_model_noise_strength_input: bool = False,
) -> nn.Module:

    model.to(device)
    model.train()

    total_loss = 0.

    print(f'Epoch: {epoch}')
    for (x, y), _ in tqdm(train_loader, 'Training denoising model'):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)
        
        # perturb data
        noise = noise_generator.generate_noise(x.shape[0], x.shape[-1], device)
        noise_coeff = torch.rand(x.shape[0], 1, 1, 1).to(device) * noise_max_strength
        x_perturbed = (1-noise_coeff)*x + noise_coeff * noise
        
        # use network to remove noise
        if mock_model_noise_strength_input:
            noise_coeff = torch.zeros_like(noise_coeff)
        denoised_x = model(x_perturbed, noise_coeff.view(-1, 1), None)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(denoised_x, x)
        total_loss += loss.item()
        loss.backward()
        
        optimizer.step()

    total_loss /= len(train_loader)
    print(f'Train loss: {total_loss}')
    print()
    return total_loss


def test_denoising_model(
    model: nn.Module,
    noise_generator: NoiseGenerator,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    noise_max_strength: float = 0.1,
    mock_model_noise_strength_input: bool = False,
) -> nn.Module:

    model.to(device)
    model.eval()

    total_loss = 0.
    total_ref_loss = 0.

    print(f'Epoch: {epoch}')
    for (x, y), _ in tqdm(test_loader, 'Testing denoising model'):
        x = x.to(device)
        y = y.to(device)
        
        # perturb data
        noise = noise_generator.generate_noise(x.shape[0], x.shape[-1], device)
        noise_coeff = torch.rand(x.shape[0], 1, 1, 1).to(device) * noise_max_strength
        x_perturbed = (1-noise_coeff)*x + noise_coeff * noise
        
        # use network to remove noise
        if mock_model_noise_strength_input:
            noise_coeff = torch.zeros_like(noise_coeff)
        denoised_x = model(x_perturbed, noise_coeff.view(-1, 1), None)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(denoised_x, x)
        ref_loss = F.mse_loss(x_perturbed, x)
        total_loss += loss.item()
        total_ref_loss += ref_loss.item()

    total_loss /= len(test_loader)
    total_ref_loss /= len(test_loader)
    print(f'Test loss: {total_loss}, Reference loss: {total_ref_loss}')
    print()
    return total_loss, total_ref_loss


def train_denoising_param_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> nn.Module:

    model.to(device)
    model.train()

    total_loss = 0.

    print(f'Epoch: {epoch}')
    for (x, x_perturbed), _ in tqdm(train_loader, 'Training denoising model'):
        optimizer.zero_grad()
        x = x.to(device)
        x_perturbed = x_perturbed.to(device)

        noise_coeff = torch.zeros(x.shape[0], 1).to(device)
        denoised_x = model(x_perturbed, noise_coeff, None)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(denoised_x, x)
        total_loss += loss.item()
        loss.backward()
        
        optimizer.step()

    total_loss /= len(train_loader)
    print(f'Train loss: {total_loss}')
    print()
    return total_loss


def test_denoising_param_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> nn.Module:

    model.to(device)
    model.eval()

    total_loss = 0.
    total_ref_loss = 0.

    print(f'Epoch: {epoch}')
    for (x, x_perturbed), _ in tqdm(test_loader, 'Testing denoising model'):
        x = x.to(device)
        x_perturbed = x_perturbed.to(device)
        
        noise_coeff = torch.zeros(x.shape[0], 1).to(device)
        denoised_x = model(x_perturbed, noise_coeff, None)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(denoised_x, x)
        ref_loss = F.mse_loss(x_perturbed, x)
        total_loss += loss.item()
        total_ref_loss += ref_loss.item()

    total_loss /= len(test_loader)
    total_ref_loss /= len(test_loader)
    print(f'Test loss: {total_loss}, Reference loss: {total_ref_loss}')
    print()
    return total_loss, total_ref_loss