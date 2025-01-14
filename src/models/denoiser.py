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


def diffusion_like_sample(
    model: nn.Module,
    generate_noisy_sample_fn: t.Callable,
    n_steps: int = 100,
    std: float = 0.2,
    device: torch.device = torch.device('cpu'),
    first_noisy_sample: t.Optional[torch.Tensor] = None
):
    noise_amplitudes = torch.normal(0., std, size=(n_steps,)).abs().sort(descending=True)
    noise_amplitudes = torch.cat((torch.tensor([1.]), noise_amplitudes.values))
    for i, noise_amplitude in tqdm(enumerate(noise_amplitudes), desc="Sampling..."):
        if i==0:
            if first_noisy_sample is not None:
                noisy_sample = first_noisy_sample
            else:
                noisy_sample = generate_noisy_sample_fn()
        else:
            noise = generate_noisy_sample_fn()
            noisy_sample = (1 - noise_amplitude) * denoised_sample + (noise_amplitude) * noise

        t = torch.full((noisy_sample.shape[0], 1), noise_amplitude).to(device)
        noisy_sample = noisy_sample.to(device)
        denoised_sample = model(noisy_sample, t, None)
    return denoised_sample


def train_denoising_conductance_param_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    h_denormalize: t.Callable,
    cmap_normalize: t.Callable,
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

        torch_h = h_denormalize(torch_h)
        mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(torch_h[:, 0], torch_h[:, 1]))
        predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
        predicted_cmap = cmap_normalize(predicted_cmap)
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
    h_denormalize: t.Callable,
    cmap_normalize: t.Callable,
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

            torch_h = h_denormalize(torch_h)
            mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(torch_h[:, 0], torch_h[:, 1]))
            predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
            predicted_cmap = cmap_normalize(predicted_cmap)
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


def train_denoising_conductance_2models(
    model_c2h: nn.Module,
    model_h2c: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer_c2h: torch.optim.Optimizer,
    optimizer_h2c: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    h_denormalize: t.Callable,
    cmap_normalize: t.Callable,
    cmap_config: t.Dict[str, t.Any],
    model_prediction: str = 'hamiltonian',
) -> nn.Module:

    model_c2h.to(device)
    model_c2h.train()

    model_h2c.to(device)
    model_h2c.train()

    total_loss_c2h = 0.
    total_loss_h2c = 0.

    num_channels = len(cmap_config['cmap_list'])

    print(f'Epoch: {epoch}')
    for (h, x, h_perturbed, x_perturbed), _ in tqdm(train_loader, 'Training denoising model'):
        x = x.to(device).squeeze(1)
        x_perturbed = x_perturbed.to(device)
        h = h.to(device)
        h_perturbed = h_perturbed.to(device)
        noise_coeff = torch.zeros(x.shape[0], 1).to(device)
        
        # Training C2H model
        optimizer_c2h.zero_grad()

        torch_h = model_c2h(x_perturbed, noise_coeff, None)
        if model_prediction == 'hamiltonian_improvement_mask':
            multiplication_factor = torch.abs(torch_h)
            torch_h = h_perturbed + multiplication_factor * h_perturbed

        predicted_x = model_h2c(torch_h, noise_coeff, None)

        loss_c2h = F.mse_loss(predicted_x, x[:, :num_channels])
        total_loss_c2h += loss_c2h.item()
        loss_c2h.backward()
        optimizer_c2h.step()

        # Training H2C model
        optimizer_h2c.zero_grad()

        # Unperturbed data
        predicted_x = model_h2c(h, noise_coeff, None)
        
        # # Perturbed data
        predicted_x_perturbed = model_h2c(h_perturbed, noise_coeff, None)

        # C2H output
        predicted_x_from_c2h = model_h2c(torch_h.detach(), noise_coeff, None)
        
        torch_h = h_denormalize(torch_h.detach())
        mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(torch_h[:, 0], torch_h[:, 1]))
        predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
        predicted_cmap = cmap_normalize(predicted_cmap).detach()

        loss_h2c = (
            (F.mse_loss(predicted_x, x[:, :num_channels]) +
            F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels]) +
            F.mse_loss(predicted_x_from_c2h, predicted_cmap[:, :num_channels])
            ) / 3
        )
        total_loss_h2c += loss_h2c.item()
        loss_h2c.backward()
        optimizer_h2c.step()


    total_loss_c2h /= len(train_loader)
    total_loss_h2c /= len(train_loader)
    print(f'Train loss C2H: {total_loss_c2h}, Train loss H2C: {total_loss_h2c}')
    print()    
    return total_loss_c2h, total_loss_h2c


def test_denoising_conductance_2models(
    model_c2h: nn.Module,
    model_h2c: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    h_denormalize: t.Callable,
    cmap_normalize: t.Callable,
    cmap_config: t.Dict[str, t.Any],
    model_prediction: str = 'hamiltonian',
) -> nn.Module:

    model_c2h.to(device)
    model_c2h.eval()

    model_h2c.to(device)
    model_h2c.eval()

    total_loss_c2h = 0.
    total_loss_h2c = 0.
    total_loss_h2c_on_c2h = 0.

    num_channels = len(cmap_config['cmap_list'])

    with torch.no_grad():
        print(f'Epoch: {epoch}')
        for (h, x, h_perturbed, x_perturbed), _ in tqdm(test_loader, 'Testing denoising model'):
            x = x.to(device).squeeze(1)
            x_perturbed = x_perturbed.to(device)
            h = h.to(device)
            h_perturbed = h_perturbed.to(device)
            noise_coeff = torch.zeros(x.shape[0], 1).to(device)

            # Testing H2C model
            # Unperturbed data
            predicted_x = model_h2c(h, noise_coeff, None)
            # Perturbed data
            predicted_x_perturbed = model_h2c(h_perturbed, noise_coeff, None)
            loss_h2c = (F.mse_loss(predicted_x, x[:, :num_channels]) + F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels])) / 2
            total_loss_h2c += loss_h2c.item()
            
            # Testing C2H model
            torch_h = model_c2h(x_perturbed, noise_coeff, None)
            if model_prediction == 'hamiltonian_improvement_mask':
                multiplication_factor = torch.abs(torch_h)
                torch_h = h_perturbed + multiplication_factor * h_perturbed

            predicted_x = model_h2c(torch_h, noise_coeff, None)
            loss_c2h = F.mse_loss(predicted_x, x[:, :num_channels])
            total_loss_c2h += loss_c2h.item()

            # Testing H2C on C2H output
            torch_h = h_denormalize(torch_h)
            mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(torch_h[:, 0], torch_h[:, 1]))
            predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
            predicted_cmap = cmap_normalize(predicted_cmap)

            loss_h2c_on_c2h = F.mse_loss(predicted_cmap[:, :num_channels], predicted_x)
            total_loss_h2c_on_c2h += loss_h2c_on_c2h.item()

    total_loss_c2h /= len(test_loader)
    total_loss_h2c /= len(test_loader)
    total_loss_h2c_on_c2h /= len(test_loader)
    print(f'Test loss C2H: {total_loss_c2h}, Test loss H2C: {total_loss_h2c}, Test loss H2C on C2H: {total_loss_h2c_on_c2h}')
    print()
    return total_loss_c2h, total_loss_h2c, total_loss_h2c_on_c2h