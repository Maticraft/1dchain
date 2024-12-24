from abc import abstractmethod, ABC
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.hamiltonian.conductance import generate_conductance_tensor
from src.models.noise_generatiron import NoiseGenerator
from src.hamiltonian.quantum_dots_chain import AtomicUnits
from src.hamiltonian.hamiltonian import transform_majorana_plus_minus_up_down_representation_to_default
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
    for (x, x_perturbed), (_, noise_amp) in tqdm(train_loader, 'Training denoising model'):
        optimizer.zero_grad()
        x = x.to(device)
        x_perturbed = x_perturbed.to(device)

        noise_coeff = noise_amp.to(device).unsqueeze(-1)
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
    reference_loss: bool = True,
) -> nn.Module:

    model.to(device)
    model.eval()

    total_loss = 0.
    total_ref_loss = 0.

    with torch.no_grad():
        print(f'Epoch: {epoch}')
        for (x, x_perturbed), (_, noise_amp) in tqdm(test_loader, 'Testing denoising model'):
            x = x.to(device)
            x_perturbed = x_perturbed.to(device)
            
            noise_coeff = noise_amp.to(device).unsqueeze(-1)
            denoised_x = model(x_perturbed, noise_coeff, None)
            
            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(denoised_x, x)
            total_loss += loss.item()
            if reference_loss:
                ref_loss = F.mse_loss(x_perturbed, x)
                total_ref_loss += ref_loss.item()

    total_loss /= len(test_loader)
    total_ref_loss /= len(test_loader)
    print(f'Test loss: {total_loss}, Reference loss: {total_ref_loss}')
    print()
    return total_loss, total_ref_loss


def train_denoising_conductance_param_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    denormalize: t.Callable,
    cmap_config: t.Dict[str, t.Any],
    model_prediction: str = 'hamiltonian',
    use_input_as_target: bool = False,
    weighting_threhsold: t.Optional[float] = None
) -> nn.Module:

    model.to(device)
    model.train()

    total_loss = 0.

    print(f'Epoch: {epoch}')
    for (h, x, h_perturbed, x_perturbed), _ in tqdm(train_loader, 'Training denoising model'):
        optimizer.zero_grad()
        x = x.to(device).squeeze(1)
        x_perturbed = x_perturbed.to(device)

        noise_coeff = torch.zeros(x.shape[0], 1).to(device)
        torch_h = model(x_perturbed, noise_coeff, None)
        if model_prediction == 'hamiltonian_improvement_mask':
            h_perturbed = h_perturbed.to(device)
            multiplication_factor = torch.abs(torch_h)
            torch_h = h_perturbed + multiplication_factor * h_perturbed

        torch_h = denormalize(torch_h)
        mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(torch_h[:, 0], torch_h[:, 1]))
        predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
        # loss is mean squared error between the predicted and true noise
        if use_input_as_target:
            target = x_perturbed.squeeze(1)
        else:
            target = x
        
        if weighting_threhsold is not None:
            mask = torch.abs(target) > weighting_threhsold
            multiplicator = torch.abs(target).max() / torch.abs(target)
            weights = torch.where(mask, multiplicator, torch.ones_like(target)).detach()
            loss = F.mse_loss(predicted_cmap, target, reduction='none')
            loss = (loss * weights).mean()
        else:
            loss = F.mse_loss(predicted_cmap, target)
        total_loss += loss.item()
        loss.backward()
        
        optimizer.step()

    total_loss /= len(train_loader)
    print(f'Train loss: {total_loss}')
    print()
    return total_loss

def test_denoising_conductance_param_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    denormalize: t.Callable,
    cmap_config: t.Dict[str, t.Any],
    reference_loss: bool = True,
    model_prediction: str = 'hamiltonian'
) -> nn.Module:

    model.to(device)
    model.eval()

    total_loss = 0.
    total_ref_loss = 0.

    with torch.no_grad():
        print(f'Epoch: {epoch}')
        for (h, x, h_perturbed, x_perturbed), _ in tqdm(test_loader, 'Testing denoising model'):
            x = x.to(device).squeeze(1)
            x_perturbed = x_perturbed.to(device)
            
            noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            torch_h = model(x_perturbed, noise_coeff, None)

            if model_prediction == 'hamiltonian_improvement_mask':
                h_perturbed = h_perturbed.to(device)
                multiplication_factor = torch.abs(torch_h)
                torch_h = h_perturbed + multiplication_factor * h_perturbed

            torch_h = denormalize(torch_h)
            mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(torch_h[:, 0], torch_h[:, 1]))
            predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(predicted_cmap, x)
        total_loss += loss.item()
        if reference_loss:
            ref_loss = F.mse_loss(x_perturbed.squeeze(1), x)
            total_ref_loss += ref_loss.item()

    total_loss /= len(test_loader)
    total_ref_loss /= len(test_loader)
    print(f'Test loss: {total_loss}, Reference loss: {total_ref_loss}')
    print()
    return total_loss, total_ref_loss