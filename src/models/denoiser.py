from abc import abstractmethod, ABC
from copy import deepcopy
import typing as t

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.hamiltonian.conductance import generate_conductance_tensor
from src.hamiltonian.hamiltonian_torch_handlers import HamiltonianConverter
from src.models.noise_generatiron import NoiseGenerator
from src.hamiltonian.hamiltonian import transform_majorana_plus_minus_up_down_representation_to_default
from src.models.diffusion_transformer import DiT
from src.models.utils import deep_update, deep_copy, weighted_update, ParameterWeightedUpdate

 
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


def train_conductance_model(
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
    for (h, x), _ in tqdm(train_loader, 'Training conductance model'):
        optimizer.zero_grad()
        x = x.to(device)
        h = h.to(device)

        noise_coeff = torch.zeros(1, 1).to(device)
        pred_h = model(x, noise_coeff, None)
        
        # loss is mean squared error between the predicted and true noise
        loss = F.mse_loss(pred_h, h)
        total_loss += loss.item()
        loss.backward()
        
        optimizer.step()

    total_loss /= len(train_loader)
    print(f'Train loss: {total_loss}')
    print()
    return total_loss


def test_conductance_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
) -> nn.Module:

    model.to(device)
    model.eval()

    total_loss = 0.
    total_ref_loss = 0.

    with torch.no_grad():
        print(f'Epoch: {epoch}')
        for (h, x), _ in tqdm(test_loader, 'Testing conductance model'):
            x = x.to(device)
            h = h.to(device)
            
            noise_coeff = torch.zeros(1, 1).to(device)
            pred_h = model(x, noise_coeff, None)

            # loss is mean squared error between the predicted and true noise
            loss = F.mse_loss(pred_h, h)
            total_loss += loss.item()

    total_loss /= len(test_loader)
    total_ref_loss /= len(test_loader)
    print(f'Test loss: {total_loss}')
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
    for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(train_loader, 'Training denoising model'):
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
        for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(test_loader, 'Testing denoising model'):
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
            multiplication_factor = torch_h
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


def train_denoising_conductance_3models(
    model_c2h: nn.Module,
    model_h2c: nn.Module,
    model_h2h: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer_c2h: torch.optim.Optimizer,
    optimizer_h2c: torch.optim.Optimizer,
    optimizer_h2h: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    h_denormalize: t.Callable,
    cmap_normalize: t.Callable,
    cmap_config: t.Dict[str, t.Any],
    model_prediction: str = 'hamiltonian',
    h2h_start_epoch: int = 0,
    train_c2h: bool = True,
    train_h2c: bool = True,
    h2c_repetitions: int = 1,
) -> nn.Module:

    model_c2h.to(device)
    model_c2h.train()

    model_h2c.to(device)
    model_h2c.train()

    model_h2h.to(device)
    model_h2h.train()

    total_loss_c2h = 0.
    total_loss_h2c = 0.
    total_loss_h2h = 0.

    num_channels = len(cmap_config['cmap_list'])

    print(f'Epoch: {epoch}')
    for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(train_loader, 'Training denoising model'):
        x = x.to(device).squeeze(1)
        x_perturbed = x_perturbed.to(device)
        h = h.to(device)
        h_perturbed = h_perturbed.to(device)
        mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)

        # Training C2H model
        if train_c2h:
            optimizer_c2h.zero_grad()

            torch_h = model_c2h(x_perturbed, mock_noise_coeff, None)
            predicted_x = model_h2c(torch_h, mock_noise_coeff, None)

            loss_c2h = F.mse_loss(predicted_x, x_perturbed[:, :num_channels])
            total_loss_c2h += loss_c2h.item()
            loss_c2h.backward()
            optimizer_c2h.step()
        else:
            with torch.no_grad():
                torch_h = model_c2h(x_perturbed, mock_noise_coeff, None)

        # Training H2H model
        if epoch >= h2h_start_epoch:
            optimizer_h2h.zero_grad()

            noise_coeff_real = noise_amp.to(device).unsqueeze(-1)
            improved_h = model_h2h(torch_h.detach(), noise_coeff_real, None)
            if model_prediction == 'hamiltonian_improvement_mask':
                multiplication_factor = abs(improved_h).clone()
                improved_h = torch.where(multiplication_factor > 0, multiplication_factor * torch_h.detach(), torch_h.detach())

            predicted_improved_x = model_h2c(improved_h, mock_noise_coeff, None)

            # threshold_low = predicted_improved_x.shape[-2] // 2 - int(predicted_improved_x.shape[-2] * 0.05)
            # threshold_high = predicted_improved_x.shape[-2] // 2 + int(predicted_improved_x.shape[-2] * 0.05)
            # improved_x_inc_loss = predicted_improved_x[..., threshold_low:threshold_high, :].mean()
            # improved_x_dec_loss = predicted_improved_x[..., :threshold_low, :].mean() + predicted_improved_x[..., threshold_high:, :].mean()
            # loss_h2h = -improved_x_inc_loss + improved_x_dec_loss
            
            loss_h2h = F.mse_loss(predicted_improved_x, x[:, :num_channels], reduction='mean') # was none
            
            # half_weights = torch.arange(0, loss_h2h.shape[-1] // 2) / (loss_h2h.shape[-1] // 2)
            # weights = torch.cat((half_weights, half_weights.flip(0))).to(device).view(1, 1, -1, 1).expand_as(loss_h2h)
            # loss_h2h = (loss_h2h * weights).mean()

            total_loss_h2h += loss_h2h.item()
            loss_h2h.backward()
            optimizer_h2h.step()


        # Training H2C model
        if train_h2c:
            optimizer_h2c.zero_grad()

            # Unperturbed data
            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            predicted_x = model_h2c(h, mock_noise_coeff, None)
            
            # # Perturbed data
            predicted_x_perturbed = model_h2c(h_perturbed, mock_noise_coeff, None)

            # C2H output
            predicted_x_from_c2h = model_h2c(torch_h.detach(), mock_noise_coeff, None)
            
            torch_h_denom = h_denormalize(torch_h.detach())
            mapped_h_predicted_c2h = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(torch_h_denom[:, 0], torch_h_denom[:, 1]))
            predicted_cmap_c2h = generate_conductance_tensor(mapped_h_predicted_c2h, cmap_config)
            predicted_cmap_c2h = cmap_normalize(predicted_cmap_c2h).detach()

            # H2H output
            if epoch >= h2h_start_epoch:
                predicted_x_from_h2h = model_h2c(improved_h.detach(), mock_noise_coeff, None)

                improved_h_denom = h_denormalize(improved_h.detach())
                mapped_h_predicted_h2h = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(improved_h_denom[:, 0], improved_h_denom[:, 1]))
                predicted_cmap_h2h = generate_conductance_tensor(mapped_h_predicted_h2h, cmap_config)
                # predicted_cmap_h2h = torch.clip(predicted_cmap_h2h, -2, 2) # why the peaks exist? hamiltonian tensors looks ok (and similar in case of c2h and h2h)

                predicted_cmap_h2h = cmap_normalize(predicted_cmap_h2h).detach()


                loss_h2c = (
                    (0.5*F.mse_loss(predicted_x, x[:, :num_channels]) +
                    0.5*F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels]) +
                    1*F.mse_loss(predicted_x_from_c2h, predicted_cmap_c2h[:, :num_channels]) +
                    2*F.mse_loss(predicted_x_from_h2h, predicted_cmap_h2h[:, :num_channels])
                    ) / 4
                )
            else:
                loss_h2c = (
                    (F.mse_loss(predicted_x, x[:, :num_channels]) +
                    F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels]) +
                    F.mse_loss(predicted_x_from_c2h, predicted_cmap_c2h[:, :num_channels])
                    ) / 3
                )

            total_loss_h2c += loss_h2c.item()
            loss_h2c.backward()
            optimizer_h2c.step()

    total_loss_c2h /= len(train_loader)
    total_loss_h2c /= len(train_loader)
    total_loss_h2h /= len(train_loader)
    print(f'Train loss C2H: {total_loss_c2h}, Train loss H2C: {total_loss_h2c}, Train loss H2H: {total_loss_h2h}')
    print()    
    return total_loss_c2h, total_loss_h2c, total_loss_h2h


def test_denoising_conductance_3models(
    model_c2h: nn.Module,
    model_h2c: nn.Module,
    model_h2h: nn.Module,
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

    model_h2h.to(device)
    model_h2h.eval()

    total_loss_c2h = 0.
    total_loss_h2c = 0.
    total_loss_h2c_on_c2h = 0.
    total_loss_h2h = 0.
    total_loss_h2c_on_h2h = 0.

    num_channels = len(cmap_config['cmap_list'])

    with torch.no_grad():
        print(f'Epoch: {epoch}')
        for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(test_loader, 'Testing denoising model'):
            x = x.to(device).squeeze(1)
            x_perturbed = x_perturbed.to(device)
            h = h.to(device)
            h_perturbed = h_perturbed.to(device)

            # Testing H2C model
            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            # Unperturbed data
            predicted_x = model_h2c(h, mock_noise_coeff, None)
            # Perturbed data
            predicted_x_perturbed = model_h2c(h_perturbed, mock_noise_coeff, None)
            loss_h2c = (F.mse_loss(predicted_x, x[:, :num_channels]) + F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels])) / 2
            total_loss_h2c += loss_h2c.item()
            
            # Testing C2H model
            torch_h = model_c2h(x_perturbed, mock_noise_coeff, None)

            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            predicted_x = model_h2c(torch_h, mock_noise_coeff, None)
            loss_c2h = F.mse_loss(predicted_x, x_perturbed[:, :num_channels])
            total_loss_c2h += loss_c2h.item()

            # Testing H2C on C2H output
            torch_h_denorm = h_denormalize(torch_h)
            mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(torch_h_denorm[:, 0], torch_h_denorm[:, 1]))
            predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
            predicted_cmap = cmap_normalize(predicted_cmap)

            loss_h2c_on_c2h = F.mse_loss(predicted_cmap[:, :num_channels], predicted_x)
            total_loss_h2c_on_c2h += loss_h2c_on_c2h.item()

            # Testing H2H model
            real_noise_coeff = noise_amp.to(device).unsqueeze(-1)
            improved_h = model_h2h(torch_h, real_noise_coeff, None)
            if model_prediction == 'hamiltonian_improvement_mask':
                multiplication_factor = abs(improved_h).clone()
                improved_h = torch.where(multiplication_factor > 0, multiplication_factor * torch_h.detach(), torch_h.detach())

            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            predicted_improved_x = model_h2c(improved_h, mock_noise_coeff, None)

            loss_h2h = F.mse_loss(predicted_improved_x, x[:, :num_channels], reduction='none').mean()
            # half_weights = torch.arange(0, loss_h2h.shape[-1] // 2) / (loss_h2h.shape[-1] // 2)
            # weights = torch.cat((half_weights, half_weights.flip(0))).to(device).view(1, 1, -1, 1).expand_as(loss_h2h)
            # loss_h2h = (loss_h2h * weights).mean()

            total_loss_h2h += loss_h2h.item()

            # Testing H2C on H2H output
            improved_h_denorm = h_denormalize(improved_h)
            mapped_h_predicted = transform_majorana_plus_minus_up_down_representation_to_default(torch.complex(improved_h_denorm[:, 0], improved_h_denorm[:, 1]))
            predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
            predicted_cmap = cmap_normalize(predicted_cmap)

            loss_h2c_on_h2h = F.mse_loss(predicted_cmap[:, :num_channels], predicted_improved_x)
            total_loss_h2c_on_h2h += loss_h2c_on_h2h.item()

    total_loss_c2h /= len(test_loader)
    total_loss_h2c /= len(test_loader)
    total_loss_h2c_on_c2h /= len(test_loader)
    total_loss_h2h /= len(test_loader)
    total_loss_h2c_on_h2h /= len(test_loader)
    print(f'Test loss C2H: {total_loss_c2h}, Test loss H2C: {total_loss_h2c}, Test loss H2C on C2H: {total_loss_h2c_on_c2h}, Test loss H2H: {total_loss_h2h}, Test loss H2C on H2H: {total_loss_h2c_on_h2h}')
    print()
    return total_loss_c2h, total_loss_h2c, total_loss_h2c_on_c2h, total_loss_h2h, total_loss_h2c_on_h2h


def train_denoising_conductance_3models_h_improvement(
    model_c2h: DiT,
    model_h2c: DiT,
    model_h2h: DiT,
    train_loader: torch.utils.data.DataLoader,
    optimizer_c2h: torch.optim.Optimizer,
    optimizer_h2c: torch.optim.Optimizer,
    optimizer_h2h: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    h_denormalize: t.Callable,
    cmap_normalize: t.Callable,
    cmap_config: t.Dict[str, t.Any],
    h2h_start_epoch: int = 0,
    train_c2h: bool = True,
    train_h2c: bool = True,
):

    model_c2h.to(device)
    model_c2h.train()

    model_h2c.to(device)
    model_h2c.train()

    model_h2h.to(device)
    model_h2h.train()

    total_loss_c2h = 0.
    total_loss_h2c = 0.
    total_loss_h2h = 0.

    num_channels = len(cmap_config['cmap_list'])

    print(f'Epoch: {epoch}')
    for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(train_loader, 'Training denoising model'):
        x = x.to(device).squeeze(1)
        x_perturbed = x_perturbed.to(device)
        h = h.to(device)
        h_perturbed = h_perturbed.to(device)
        mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)

        # Training C2H model
        if train_c2h:
            optimizer_c2h.zero_grad()

            # torch_h = model_c2h(x_perturbed, mock_noise_coeff, None)
            # predicted_x_1 = model_h2c(torch_h, mock_noise_coeff, None)

            params_map = model_c2h(x_perturbed, mock_noise_coeff, None, matrix_output=False)
            predicted_x = model_h2c(params_map, mock_noise_coeff, None, matrix_input=False)

            loss_c2h = F.mse_loss(predicted_x, x_perturbed[:, :num_channels])
            total_loss_c2h += loss_c2h.item()
            loss_c2h.backward()
            optimizer_c2h.step()

        # Training H2H model
        if epoch >= h2h_start_epoch:
            optimizer_h2h.zero_grad()

            params_map = model_c2h(x_perturbed, mock_noise_coeff, None, matrix_output=False)
            # params_map = deep_copy(params_map, detach=True) # why it makes c2h model not training?

            noise_coeff_real = noise_amp.to(device).unsqueeze(-1)
            improve_map = model_h2h(params_map, noise_coeff_real, None, matrix_input=False, matrix_output=False)

            improved_params_map = deep_update(params_map, improve_map, detach=False)

            predicted_improved_x = model_h2c(improved_params_map, mock_noise_coeff, None, matrix_input=False)

            # threshold_low = predicted_improved_x.shape[-2] // 2 - int(predicted_improved_x.shape[-2] * 0.05)
            # threshold_high = predicted_improved_x.shape[-2] // 2 + int(predicted_improved_x.shape[-2] * 0.05)
            # improved_x_inc_loss = predicted_improved_x[..., threshold_low:threshold_high, :].mean()
            # improved_x_dec_loss = predicted_improved_x[..., :threshold_low, :].mean() + predicted_improved_x[..., threshold_high:, :].mean()
            # loss_h2h = -improved_x_inc_loss + improved_x_dec_loss
            
            loss_h2h = F.mse_loss(predicted_improved_x, x[:, :num_channels], reduction='mean') # was none
            
            # half_weights = torch.arange(0, loss_h2h.shape[-1] // 2) / (loss_h2h.shape[-1] // 2)
            # weights = torch.cat((half_weights, half_weights.flip(0))).to(device).view(1, 1, -1, 1).expand_as(loss_h2h)
            # loss_h2h = (loss_h2h * weights).mean()

            total_loss_h2h += loss_h2h.item()
            loss_h2h.backward()
            optimizer_h2h.step()


        # Training H2C model
        if train_h2c:
            optimizer_h2c.zero_grad()

            # Unperturbed data
            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            predicted_x = model_h2c(h, mock_noise_coeff, None)
            
            # # Perturbed data
            predicted_x_perturbed = model_h2c(h_perturbed, mock_noise_coeff, None)

            # C2H output
            predicted_h = model_c2h(x_perturbed, mock_noise_coeff, None)
            predicted_x_from_c2h = model_h2c(predicted_h.detach(), mock_noise_coeff, None)
            
            torch_h_denom = h_denormalize(predicted_h.detach())
            mapped_h_predicted_c2h = torch.complex(torch_h_denom[:, 0], torch_h_denom[:, 1])
            predicted_cmap_c2h = generate_conductance_tensor(mapped_h_predicted_c2h, cmap_config)
            predicted_cmap_c2h = cmap_normalize(predicted_cmap_c2h).detach()

            # H2H output
            if epoch >= h2h_start_epoch:
                improved_params_map = deep_copy(improved_params_map, detach=True)

                on_site_params, inter_site_params = model_h2c.hamiltonian_embeddder.hamiltonian_extractor.from_params_map(improved_params_map)
                improved_h = model_c2h.hamiltonian_unpatch.hamiltonian_constructor(on_site_params, inter_site_params)

                predicted_x_from_h2h = model_h2c(improved_params_map, mock_noise_coeff, None, matrix_input=False)

                improved_h_denom = h_denormalize(improved_h.detach())
                mapped_h_predicted_h2h = torch.complex(improved_h_denom[:, 0], improved_h_denom[:, 1])
                predicted_cmap_h2h = generate_conductance_tensor(mapped_h_predicted_h2h, cmap_config)
                # predicted_cmap_h2h = torch.clip(predicted_cmap_h2h, -2, 2) # why the peaks exist? hamiltonian tensors looks ok (and similar in case of c2h and h2h)

                predicted_cmap_h2h = cmap_normalize(predicted_cmap_h2h).detach()

                loss_h2c = (
                    (0.5*F.mse_loss(predicted_x, x[:, :num_channels]) +
                    0.5*F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels]) +
                    1*F.mse_loss(predicted_x_from_c2h, predicted_cmap_c2h[:, :num_channels]) +
                    2*F.mse_loss(predicted_x_from_h2h, predicted_cmap_h2h[:, :num_channels])
                    ) / 4
                )
            else:
                loss_h2c = (
                    (F.mse_loss(predicted_x, x[:, :num_channels]) +
                    F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels]) +
                    F.mse_loss(predicted_x_from_c2h, predicted_cmap_c2h[:, :num_channels])
                    ) / 3
                )

            total_loss_h2c += loss_h2c.item()
            loss_h2c.backward()
            optimizer_h2c.step()

    total_loss_c2h /= len(train_loader)
    total_loss_h2c /= len(train_loader)
    total_loss_h2h /= len(train_loader)
    print(f'Train loss C2H: {total_loss_c2h}, Train loss H2C: {total_loss_h2c}, Train loss H2H: {total_loss_h2h}')
    print()    
    return total_loss_c2h, total_loss_h2c, total_loss_h2h


def test_denoising_conductance_3models_h_improvement(
    model_c2h: DiT,
    model_h2c: DiT,
    model_h2h: DiT,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    h_denormalize: t.Callable,
    cmap_normalize: t.Callable,
    cmap_config: t.Dict[str, t.Any],
):

    model_c2h.to(device)
    model_c2h.eval()

    model_h2c.to(device)
    model_h2c.eval()

    model_h2h.to(device)
    model_h2h.eval()

    total_loss_c2h = 0.
    total_loss_h2c = 0.
    total_loss_h2c_on_c2h = 0.
    total_loss_h2h = 0.
    total_loss_h2c_on_h2h = 0.

    num_channels = len(cmap_config['cmap_list'])

    with torch.no_grad():
        print(f'Epoch: {epoch}')
        for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(test_loader, 'Testing denoising model'):
            x = x.to(device).squeeze(1)
            x_perturbed = x_perturbed.to(device)
            h = h.to(device)
            h_perturbed = h_perturbed.to(device)

            # Testing H2C model
            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            # Unperturbed data
            predicted_x = model_h2c(h, mock_noise_coeff, None)
            
            # Perturbed data
            predicted_x_perturbed = model_h2c(h_perturbed, mock_noise_coeff, None)
            loss_h2c = (F.mse_loss(predicted_x, x[:, :num_channels]) + F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels])) / 2
            total_loss_h2c += loss_h2c.item()
            
            # Testing C2H model
            torch_h = model_c2h(x_perturbed, mock_noise_coeff, None)

            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            predicted_x = model_h2c(torch_h, mock_noise_coeff, None)
            loss_c2h = F.mse_loss(predicted_x, x_perturbed[:, :num_channels])
            total_loss_c2h += loss_c2h.item()

            # Testing H2C on C2H output
            torch_h_denorm = h_denormalize(torch_h)
            mapped_h_predicted = torch.complex(torch_h_denorm[:, 0], torch_h_denorm[:, 1])
            predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
            predicted_cmap = cmap_normalize(predicted_cmap)

            loss_h2c_on_c2h = F.mse_loss(predicted_cmap[:, :num_channels], predicted_x)
            total_loss_h2c_on_c2h += loss_h2c_on_c2h.item()

            # Testing H2H model
            params_map = model_c2h.forward_to_params(x_perturbed, mock_noise_coeff, None)

            noise_coeff_real = noise_amp.to(device).unsqueeze(-1)
            improve_map = model_h2h.forward_from_params_to_params(params_map, noise_coeff_real, None)

            improved_params_map = deep_update(params_map, improve_map)

            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            predicted_improved_x = model_h2c.forward_from_params(improved_params_map, mock_noise_coeff, None)

            loss_h2h = F.mse_loss(predicted_improved_x, x[:, :num_channels], reduction='none').mean()
            # half_weights = torch.arange(0, loss_h2h.shape[-1] // 2) / (loss_h2h.shape[-1] // 2)
            # weights = torch.cat((half_weights, half_weights.flip(0))).to(device).view(1, 1, -1, 1).expand_as(loss_h2h)
            # loss_h2h = (loss_h2h * weights).mean()

            total_loss_h2h += loss_h2h.item()

            # Testing H2C on H2H output
            on_site_params, inter_site_params = model_h2c.hamiltonian_embeddder.hamiltonian_extractor.from_params_map(improved_params_map)
            improved_h = model_c2h.hamiltonian_unpatch.hamiltonian_constructor(on_site_params, inter_site_params)
                
            improved_h_denorm = h_denormalize(improved_h)
            mapped_h_predicted = torch.complex(improved_h_denorm[:, 0], improved_h_denorm[:, 1])
            predicted_cmap = generate_conductance_tensor(mapped_h_predicted, cmap_config)
            predicted_cmap = cmap_normalize(predicted_cmap)

            loss_h2c_on_h2h = F.mse_loss(predicted_cmap[:, :num_channels], predicted_improved_x)
            total_loss_h2c_on_h2h += loss_h2c_on_h2h.item()

    total_loss_c2h /= len(test_loader)
    total_loss_h2c /= len(test_loader)
    total_loss_h2c_on_c2h /= len(test_loader)
    total_loss_h2h /= len(test_loader)
    total_loss_h2c_on_h2h /= len(test_loader)
    print(f'Test loss C2H: {total_loss_c2h}, Test loss H2C: {total_loss_h2c}, Test loss H2C on C2H: {total_loss_h2c_on_c2h}, Test loss H2H: {total_loss_h2h}, Test loss H2C on H2H: {total_loss_h2c_on_h2h}')
    print()
    return total_loss_c2h, total_loss_h2c, total_loss_h2c_on_c2h, total_loss_h2h, total_loss_h2c_on_h2h


def train_denoising_conductance_2models_h_improvement(
    model_c2h: DiT,
    model_h2c: DiT,
    train_loader: torch.utils.data.DataLoader,
    optimizer_c2h: torch.optim.Optimizer,
    optimizer_h2c: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    h_denormalize: t.Callable,
    cmap_normalize: t.Callable,
    cmap_config: t.Dict[str, t.Any],
    hamiltonian_converter: HamiltonianConverter,
    train_c2h: bool = True,
    train_h2c: bool = True,
    stage2_epoch: int = 60,
):

    model_c2h.to(device)
    model_c2h.train()

    model_h2c.to(device)
    model_h2c.train()

    total_loss_c2h = 0.
    total_loss_h2c = 0.

    num_channels = len(cmap_config['cmap_list'])

    print(f'Epoch: {epoch}')
    for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(train_loader, 'Training denoising model'):
        x = x.to(device).squeeze(1)
        x_perturbed = x_perturbed.to(device)
        h = h.to(device)
        h_perturbed = h_perturbed.to(device)
        mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
        noise_coeff_real = noise_amp.to(device).unsqueeze(-1)

        h_perturbed_params_map = hamiltonian_converter.from_matrix_to_params(h_perturbed)

        # Training C2H model
        if train_c2h:
            if epoch < stage2_epoch:
                target = x_perturbed
            else:
                target = x

            optimizer_c2h.zero_grad()

            # torch_h = model_c2h(x_perturbed, mock_noise_coeff, None)
            # predicted_x_1 = model_h2c(torch_h, mock_noise_coeff, None)

            params_map = model_c2h(x_perturbed, noise_coeff_real, None, matrix_output=False)
            # improved_map = deep_update(h_perturbed_params_map, params_map, detach=False)
            # improved_map = weighted_update(h_perturbed_params_map, params_map, alpha=0.5)

            predicted_x = model_h2c(params_map, mock_noise_coeff, None, matrix_input=False)

            loss_c2h = F.mse_loss(predicted_x, target[:, :num_channels])

            # loss_c2h = F.mse_loss(params_map, h, reduction='mean') # was none
            total_loss_c2h += loss_c2h.item()
            loss_c2h.backward()

            # After loss_c2h.backward() but before optimizer_c2h.step()
            # has_grad = False
            # for p in model_c2h.parameters():
            #     if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0:
            #         has_grad = True
            #         break
            # print(f"model_c2h has gradients: {has_grad}")

            # has_grad = False
            # for p in model_h2c.parameters():
            #     if p.grad is not None and torch.sum(torch.abs(p.grad)) > 0:
            #         has_grad = True
            #         break
            # print(f"model_h2c has gradients: {has_grad}")

            optimizer_c2h.step()

        # Training H2C model
        if train_h2c:
            optimizer_h2c.zero_grad()

            # Unperturbed data
            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            predicted_x = model_h2c(h, mock_noise_coeff, None)
            
            # # Perturbed data
            predicted_x_perturbed = model_h2c(h_perturbed, mock_noise_coeff, None)

            # C2H basic output (probably not needed)
            # predicted_h = model_c2h(x_perturbed, noise_coeff_real, None)
            # predicted_x_from_c2h = model_h2c(predicted_h.detach(), mock_noise_coeff, None)
            
            # torch_h_denom = h_denormalize(predicted_h.detach())
            # mapped_h_predicted_c2h = torch.complex(torch_h_denom[:, 0], torch_h_denom[:, 1])
            # predicted_cmap_c2h = generate_conductance_tensor(mapped_h_predicted_c2h, cmap_config)
            # predicted_cmap_c2h = cmap_normalize(predicted_cmap_c2h).detach()

            # C2H improved output
            improved_params_map = deep_copy(params_map, detach=True)

            # predicted_improved_x_from_c2h = model_h2c(improved_params_map, mock_noise_coeff, None, matrix_input=False)

            improved_h = hamiltonian_converter.from_params_to_matrix(improved_params_map)
            
            improved_h_denom = h_denormalize(improved_h.detach())
            improved_h_complex = torch.complex(improved_h_denom[:, 0], improved_h_denom[:, 1])
            predicted_cmap_improved_c2h = generate_conductance_tensor(improved_h_complex, cmap_config)
            predicted_cmap_improved_c2h = cmap_normalize(predicted_cmap_improved_c2h).detach()

            predicted_improved_x_from_c2h = model_h2c(improved_h, mock_noise_coeff, None, matrix_input=True)

            loss_h2c = (
                (F.mse_loss(predicted_x, x[:, :num_channels]) +
                F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels]) +
                # F.mse_loss(predicted_x_from_c2h, predicted_cmap_c2h[:, :num_channels]) +
                F.mse_loss(predicted_improved_x_from_c2h, predicted_cmap_improved_c2h[:, :num_channels])
                ) / 4
            )

            total_loss_h2c += loss_h2c.item()
            loss_h2c.backward()
            optimizer_h2c.step()

    total_loss_c2h /= len(train_loader)
    total_loss_h2c /= len(train_loader)
    print(f'Train loss C2H: {total_loss_c2h}, Train loss H2C: {total_loss_h2c}')
    print()    
    return total_loss_c2h, total_loss_h2c


def test_denoising_conductance_2models_h_improvement(
    model_c2h: DiT,
    model_h2c: DiT,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
    h_denormalize: t.Callable,
    cmap_normalize: t.Callable,
    cmap_config: t.Dict[str, t.Any],
    hamiltonian_converter: HamiltonianConverter,
    stage2_epoch: int = 60,
):

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
        for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(test_loader, 'Testing denoising model'):
            x = x.to(device).squeeze(1)
            x_perturbed = x_perturbed.to(device)
            h = h.to(device)
            h_perturbed = h_perturbed.to(device)

            if epoch < stage2_epoch:
                target = x_perturbed
            else:
                target = x

            h_perturbed_params_map = hamiltonian_converter.from_matrix_to_params(h_perturbed)

            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            noise_coeff_real = noise_amp.to(device).unsqueeze(-1)

            # Testing H2C model
            # Unperturbed data
            predicted_x = model_h2c(h, mock_noise_coeff, None)
            
            # Perturbed data
            predicted_x_perturbed = model_h2c(h_perturbed, mock_noise_coeff, None)
            loss_h2c = (F.mse_loss(predicted_x, x[:, :num_channels]) + F.mse_loss(predicted_x_perturbed, x_perturbed[:, :num_channels])) / 2
            total_loss_h2c += loss_h2c.item()
            
            # Testing C2H model
            params_map = model_c2h(x_perturbed, noise_coeff_real, None, matrix_output=False)
            # improved_map = deep_update(h_perturbed_params_map, params_map, detach=False)
            # improved_map = weighted_update(h_perturbed_params_map, params_map, alpha=0.5)

            predicted_x = model_h2c(params_map, mock_noise_coeff, None, matrix_input=False)
            loss_c2h = F.mse_loss(predicted_x, target[:, :num_channels])
            total_loss_c2h += loss_c2h.item()

            # Testing H2C on C2H output
            torch_h = hamiltonian_converter.from_params_to_matrix(params_map)
            torch_h_denorm = h_denormalize(torch_h)
            mapped_h_predicted = torch.complex(torch_h_denorm[:, 0], torch_h_denorm[:, 1])
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


def train_denoising_conductance_DiTGEN_AE(
    generator: DiT,
    encoder: DiT,
    decoder: DiT,
    hamiltonian_converter: HamiltonianConverter,
    train_loader: torch.utils.data.DataLoader,
    optimizer_gen: torch.optim.Optimizer,
    optimizer_encoder: torch.optim.Optimizer,
    optimizer_decoder: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    train_gen: bool = True,
    train_ae: bool = True,
):

    generator.to(device)
    generator.train()

    encoder.to(device)
    encoder.train()

    decoder.to(device)
    decoder.train()

    total_loss_gen = 0.
    total_loss_ae = 0.
    total_contrastive_loss = 0.

    print(f'Epoch: {epoch}')
    for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(train_loader, 'Training denoising model'):
        x = x.to(device).squeeze(1)
        x_perturbed = x_perturbed.to(device)
        h = h.to(device)
        h_perturbed = h_perturbed.to(device)
        mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
        noise_coeff_real = noise_amp.to(device).unsqueeze(-1)
     
        h_perturbed_params_map = hamiltonian_converter.from_matrix_to_params(h_perturbed)

        # Training GEN model
        if train_gen:
            optimizer_gen.zero_grad()

            params_map = generator(x_perturbed, noise_coeff_real, None, matrix_output=False)
            
            # improved_map = deep_update(h_perturbed_params_map, params_map, detach=False)
            # improved_map = weighted_update(h_perturbed_params_map, params_map, alpha=0.5)
          
            predicted_latent = encoder(params_map, mock_noise_coeff, None, matrix_input=False)
            predicted_h = decoder(predicted_latent, mock_noise_coeff, None)

            input_h = hamiltonian_converter.from_params_to_matrix(params_map)

            target_latent = encoder(h, mock_noise_coeff, None, matrix_input=True)

            loss_gen = F.mse_loss(predicted_h, input_h) + F.mse_loss(predicted_latent, target_latent)
            total_loss_gen += loss_gen.item()
            loss_gen.backward()
            optimizer_gen.step()

        # Training AE model
        if train_ae:
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()

            predicted_latent = encoder(h, mock_noise_coeff, None)
            predicted_h = decoder(predicted_latent, mock_noise_coeff, None)

            loss_ae = F.mse_loss(predicted_h, h, reduction='mean')
            
            predicted_latent = encoder(input_h.detach(), mock_noise_coeff, None)
            predicted_h = decoder(predicted_latent, mock_noise_coeff, None)
            # contrastive_loss = -F.mse_loss(predicted_h, input_h.detach())

            total_loss_ae += loss_ae.item()
            # total_contrastive_loss += contrastive_loss.item()
            # loss_ae = loss_ae + contrastive_loss
            loss_ae.backward()
            optimizer_encoder.step()
            optimizer_decoder.step()


    total_loss_gen /= len(train_loader)
    total_loss_ae /= len(train_loader)
    print(f'Train loss GEN: {total_loss_gen}, Train loss AE: {total_loss_ae}, Train contrastive loss: {total_contrastive_loss}')
    print()    
    return total_loss_gen, total_loss_ae, total_contrastive_loss


def test_denoising_conductance_DiTGEN_AE(
    generator: DiT,
    encoder: DiT,
    decoder: DiT,
    hamiltonian_converter: HamiltonianConverter,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epoch: int,
):
    generator.to(device)
    generator.eval()

    encoder.to(device)
    encoder.eval()

    decoder.to(device)
    decoder.eval()

    total_loss_gen = 0.
    total_loss_ae = 0.
    total_loss_ae_gen = 0.

    print(f'Epoch: {epoch}')
    with torch.no_grad():
        for (h, x, h_perturbed, x_perturbed), (_, noise_amp) in tqdm(test_loader, 'Testing denoising model'):
            x = x.to(device).squeeze(1)
            x_perturbed = x_perturbed.to(device)
            h = h.to(device)
            h_perturbed = h_perturbed.to(device)

            mock_noise_coeff = torch.zeros(x.shape[0], 1).to(device)
            noise_coeff_real = noise_amp.to(device).unsqueeze(-1)

            # Testing GEN model
            params_map = generator(x_perturbed, noise_coeff_real, None, matrix_output=False)
            
            predicted_latent = encoder(params_map, mock_noise_coeff, None, matrix_input=False)

            target_latent = encoder(h, mock_noise_coeff, None, matrix_input=True)

            loss_gen = F.mse_loss(predicted_latent, target_latent)
            total_loss_gen += loss_gen.item()

            # Testing AE model
            latent = encoder(h, mock_noise_coeff, None)
            predicted_h = decoder(latent, mock_noise_coeff, None)

            loss_ae = F.mse_loss(predicted_h, h)

            total_loss_ae += loss_ae.item()

            # Testing AE on GEN output

            generated_h = generator(x_perturbed, noise_coeff_real, None, matrix_output=True)
            latent = encoder(generated_h, mock_noise_coeff, None)
            predicted_h = decoder(latent, mock_noise_coeff, None)
            loss_ae_gen = F.mse_loss(predicted_h, generated_h)
            total_loss_ae_gen += loss_ae_gen.item()

    total_loss_gen /= len(test_loader)
    total_loss_ae /= len(test_loader)
    print(f'Test loss GEN: {total_loss_gen}, Test loss AE: {total_loss_ae}, Test loss AE on GEN: {total_loss_ae_gen}')
    print()    
    return total_loss_gen, total_loss_ae, total_loss_ae_gen