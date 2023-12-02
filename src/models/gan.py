import typing as t

import torch
import torch.nn as nn
from tqdm import tqdm

from src.models.utils import generate_sample_from_mean_and_covariance
from src.torch_utils import torch_total_polarization_loss


class Discriminator(nn.Module):
    def __init__(self, model_class: t.Type[nn.Module], model_config: t.Dict[str, t.Any]):
        super(Discriminator, self).__init__()
        self.nn = model_class(**model_config)
        self.nn_out_features = model_config['representation_dim'] if isinstance(model_config['representation_dim'], int) else int(sum(model_config['representation_dim']))
        self.classifier = self._get_mlp(3, self.nn_out_features, 128, 1)

    def forward(self, x: torch.Tensor):
        out = self.nn(x)
        return self.classifier(out)

    def _get_mlp(self, layers_num: int, input_size: int, hidden_size: int, output_size: int):
        layers = []
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            elif i == layers_num - 1:
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Linear(hidden_size, output_size))
            else:
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=0.01))
        return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self, model_class: t.Type[nn.Module], **model_config: t.Dict[str, t.Any]):
        super(Generator, self).__init__()
        self.nn_in_features = model_config['representation_dim'] if isinstance(model_config['representation_dim'], int) else int(sum(model_config['representation_dim']))
        self.nn = model_class(**model_config)
        if 'distribution' in model_config:
            activation = self._create_activation_from_distribution(model_config['distribution'])
        else:
            activation = nn.LeakyReLU(negative_slope=0.01)

        self.skip_noise_converter = model_config.get('skip_noise_converter', False)
        self.noise_converter = self._get_mlp(5, self.nn_in_features, self.nn_in_features, self.nn_in_features, final_activation=activation)

    def forward(self, x: torch.Tensor):
        if not self.skip_noise_converter:
            x = self.noise_converter(x)
        return self.nn(x)

    def _create_activation_from_distribution(self, distribution: t.Tuple[torch.Tensor, torch.Tensor]):
        mu, std = distribution
        class Activation(nn.Module):
            def __init__(self, mu: torch.Tensor, std: torch.Tensor):
                super(Activation, self).__init__()
                self.mu = mu
                self.std = std
            def activation(self, x: torch.Tensor):
                mu = self.mu.to(x.device)
                std = self.std.to(x.device)
                min_cutoff = torch.maximum(x, mu - 2*std)
                max_cutoff = torch.minimum(min_cutoff, mu + 2*std)
                return max_cutoff
            def forward(self, x: torch.Tensor):
                return self.activation(x)
        return Activation(mu, std)

    def _get_mlp(self, layers_num: int, input_size: int, hidden_size: int, output_size: int, final_activation = t.Callable):
        layers = []
        if layers_num == 1:
            return nn.Linear(input_size, output_size)
        for i in range(layers_num):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            elif i == layers_num - 1:
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Linear(hidden_size, output_size))
                layers.append(final_activation)
            else:
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.LeakyReLU(negative_slope=0.01))
        return nn.Sequential(*layers)

    def get_noise(self, batch_size: int, device: torch.device, noise_type: str = 'gaussian', **kwargs: t.Dict[str, t.Any]):
        if noise_type == 'gaussian':
            return torch.randn((batch_size, self.nn_in_features), device=device)
        elif noise_type == 'uniform':
            return torch.rand((batch_size, self.nn_in_features), device=device)
        elif noise_type == 'hybrid':
            return torch.cat([torch.randn((batch_size, self.nn_in_features // 2), device=device), torch.rand((batch_size, self.nn_in_features // 2), device=device)], dim=-1)
        elif noise_type == 'custom':
            mean = kwargs['mean'].unsqueeze(0).expand(batch_size, -1)
            std = kwargs['std'].unsqueeze(0).expand(batch_size, -1)
            return torch.normal(mean, std).to(device)
        elif noise_type == 'covariance':
            mean = kwargs['mean']
            cov = kwargs['covariance']
            mean_freq = mean[:self.nn_in_features // 2]
            mean_block = mean[self.nn_in_features // 2:]
            cov_freq = cov[:self.nn_in_features // 2, :self.nn_in_features // 2]
            cov_block = cov[self.nn_in_features // 2:, self.nn_in_features // 2:]
            freq_noise = generate_sample_from_mean_and_covariance(mean_freq, cov_freq, batch_size)
            block_noise = generate_sample_from_mean_and_covariance(mean_block, cov_block, batch_size)
            return torch.cat([freq_noise, block_noise], dim=-1).to(device)
        else:
            raise ValueError('Unknown noise type')


def test_noise_controller(
    generator_model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
    correct_distribution: t.Tuple[torch.Tensor, torch.Tensor],
):
    generator_model.to(device)
    generator_model.eval()

    mean, std = correct_distribution

    total_polarization_loss = 0

    print(f'Epoch: {epoch}')
    for (x, y), _ in tqdm(test_loader, 'Training noise controller for generator'):
        x = x.to(device)

        z = generator_model.get_noise(x.shape[0], device, noise_type='hybrid')
        x_hat = generator_model(z)

        loss = torch_total_polarization_loss(x_hat)
        total_polarization_loss += loss.item()


    total_polarization_loss /= len(test_loader)

    print(f'Total classifier loss: {total_polarization_loss}\n')

    return total_polarization_loss


def train_noise_controller(
    generator_model: nn.Module,
    # encoder_model: nn.Module,
    # classifier_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
    noise_controller_optimizer: torch.optim.Optimizer,
    correct_distribution: t.Tuple[torch.Tensor, torch.Tensor],
):

    criterion = nn.BCELoss()

    # classifier_model.to(device)
    generator_model.to(device)
    # encoder_model.to(device)

    # classifier_model.train()
    generator_model.train()
    # encoder_model.train()

    mean, std = correct_distribution

    total_classifier_loss = 0
    total_ddistribution_loss = 0

    print(f'Epoch: {epoch}')
    for (x, y), _ in tqdm(train_loader, 'Training noise controller for generator'):
        x = x.to(device)
        noise_controller_optimizer.zero_grad()
        # generator_model.nn.requires_grad_(False)
        # encoder_model.requires_grad_(False)
        # classifier_model.requires_grad_(False)

        z = generator_model.get_noise(x.shape[0], device, noise_type='hybrid')
        x_hat = generator_model(z)
        # latent_prime = encoder_model(x_hat)
        # y_hat = classifier_model(x_hat)
        # desired_y = torch.ones_like(y_hat)
        # loss = criterion(y_hat, desired_y)

        loss = torch_total_polarization_loss(x_hat)
        total_classifier_loss += loss.item()

        # distrib_loss = distribution_loss(x_hat, (mean.to(device), std.to(device)))
        # total_ddistribution_loss += distrib_loss.item()
        # loss += 5*distrib_loss

        loss.backward()
        noise_controller_optimizer.step()

    total_classifier_loss /= len(train_loader)
    total_ddistribution_loss /= len(train_loader)

    print(f'Total classifier loss: {total_classifier_loss}\n')
    print(f'Total distribution loss: {total_ddistribution_loss}\n')

    return total_classifier_loss, total_ddistribution_loss


def train_gan(
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    init_distribution: t.Optional[t.Tuple[torch.Tensor, torch.Tensor]] = None,
    cov_matrix: t.Optional[torch.Tensor] = None,
    training_switch_loss_ratio: float = 1.5,
    start_training_mode: str = 'discriminator',
    data_label: t.Optional[int] = None,
):

    criterion = nn.BCEWithLogitsLoss()

    generator.to(device)
    discriminator.to(device)

    generator.train()
    discriminator.train()

    total_generator_loss = 0
    total_discriminator_loss = 0
    generator_loss = torch.tensor(0.)

    training_mode = start_training_mode
    generator.eval()
    generator.requires_grad_(False)

    print(f'Epoch: {epoch}')
    for (x, y), _ in tqdm(train_loader, 'Training model'):
        x = x.to(device)
        discriminator_optimizer.zero_grad()

        if init_distribution is not None and cov_matrix is not None:
            z = generator.get_noise(x.shape[0], device, noise_type='covariance', mean=init_distribution[0], covariance=cov_matrix)
        elif init_distribution is not None:
            z = generator.get_noise(x.shape[0], device, noise_type='custom', mean=init_distribution[0], std=init_distribution[1])
        else:
            z = generator.get_noise(x.shape[0], device, noise_type='hybrid')
        x_hat = generator(z)

        x = x if data_label is None else x[(y == data_label).squeeze()]
        if x.shape[0] > 1:
            real_prediction = discriminator(x)
        else:
            real_prediction = torch.tensor(1.)

        fake_prediction = discriminator(x_hat.detach())

        real_loss = criterion(real_prediction, torch.ones_like(real_prediction))
        fake_loss = criterion(fake_prediction, torch.zeros_like(fake_prediction))
        discriminator_loss = (real_loss + fake_loss) / 2

        total_discriminator_loss += discriminator_loss.item()

        if generator_loss.item() > training_switch_loss_ratio * discriminator_loss.item():
            generator.train()
            generator.requires_grad_(True)
            training_mode = 'generator'

        if training_mode == 'discriminator':
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()
            generator.eval()
            generator.requires_grad_(False)

        generator_optimizer.zero_grad()

        if init_distribution is not None and cov_matrix is not None:
            z2 = generator.get_noise(y.shape[0], device, noise_type='covariance', mean=init_distribution[0], covariance=cov_matrix)
        elif init_distribution is not None:
            z2 = generator.get_noise(y.shape[0], device, noise_type='custom', mean=init_distribution[0], std=init_distribution[1])
        else:
            z2 = generator.get_noise(y.shape[0], device, noise_type='hybrid')
        x_hat2 = generator(z2)
        fake_prediction2 = discriminator(x_hat2)
        generator_loss = criterion(fake_prediction2, torch.ones_like(fake_prediction2))
        total_generator_loss += generator_loss.item()

        if discriminator_loss.item() > training_switch_loss_ratio * generator_loss.item():
            generator.eval()
            generator.requires_grad_(False)
            training_mode = 'discriminator'

        if training_mode == 'generator':
            generator_loss.backward()
            generator_optimizer.step()


    total_generator_loss /= len(train_loader)
    total_discriminator_loss /= len(train_loader)

    print(f'Generator Loss: {total_generator_loss}')
    print(f'Discriminator Loss: {total_discriminator_loss}\n')

    return total_generator_loss, total_discriminator_loss, training_mode