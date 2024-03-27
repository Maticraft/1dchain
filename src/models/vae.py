from src.models.utils import diagonal_loss, edge_diff, eigenvectors_loss, kl_divergence_loss
import torch.nn as nn
from tqdm import tqdm
from src.models.positional_autoencoder import PositionalEncoder


import torch
from torch.distributions.normal import Normal


import typing as t


class VariationalPositionalEncoder(PositionalEncoder):
    def __init__(self, input_size: t.Tuple[int, int, int], representation_dim: t.Union[int, t.Tuple[int, int]], **kwargs: t.Dict[str, t.Any]):
        new_representation_dim = 2*representation_dim if isinstance(representation_dim, int) else (2*representation_dim[0], 2*representation_dim[1])
        super(VariationalPositionalEncoder, self).__init__(input_size, new_representation_dim, **kwargs)

    def forward(self, x: torch.Tensor, return_distr: bool = False, eps: float = 1e-10):
        x = super(VariationalPositionalEncoder, self).forward(x)
        freq_enc, block_enc = x[..., :self.freq_dim], x[..., self.freq_dim:]
        freq_mu, freq_std = torch.split(freq_enc, self.freq_dim // 2, dim=-1)
        block_mu, block_std = torch.split(block_enc, self.block_dim // 2, dim=-1)
        freq_dist = Normal(freq_mu, freq_std.exp() + eps)
        block_dist = Normal(block_mu, block_std.exp() + eps)
        freq_sample = freq_dist.rsample()
        block_sample = block_dist.rsample()
        sample = torch.cat([freq_sample, block_sample], dim=-1)
        if return_distr:
            return sample, (freq_dist, block_dist)
        else:
            return sample


def train_vae(
    vencoder_model: nn.Module,
    decoder_model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    device: torch.device,
    vencoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    edge_loss: bool = False,
    edge_loss_weight: float = .5,
    eigenstates_loss: bool = False,
    eigenstates_loss_weight: float = .5,
    diag_loss: bool = False,
    diag_loss_weight: float = .01,
    kl_loss: bool = True,
    kl_loss_weight: float = 0.01,
):
    criterion = nn.MSELoss()

    vencoder_model.to(device)
    decoder_model.to(device)

    vencoder_model.train()
    decoder_model.train()

    total_reconstruction_loss = 0
    total_edge_loss = 0
    total_eigenstates_loss = 0
    total_diag_loss = 0
    total_kl_loss = 0

    print(f'Epoch: {epoch}')
    for (x, _), eig_dec in tqdm(train_loader, 'Training model'):
        x = x.to(device)
        vencoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        z, (freq_dist, block_dist) = vencoder_model(x, return_distr = True)
        x_hat = decoder_model(z)
        loss = criterion(x_hat, x)
        total_reconstruction_loss += torch.mean(loss).item()

        if kl_loss:
            k1_loss = kl_divergence_loss(freq_dist).mean() + kl_divergence_loss(block_dist).mean()
            total_kl_loss += k1_loss.item()
            loss += kl_loss_weight * k1_loss

        if edge_loss:
            e_loss = edge_diff(x_hat, x, criterion, edge_width=8)
            loss += edge_loss_weight * e_loss
            total_edge_loss += torch.mean(e_loss).item()

        if eigenstates_loss:
            assert eig_dec is not None, "Incorrect eigen decomposition values"
            eig_dec = eig_dec[0].to(device), eig_dec[1].to(device)
            eig_loss = eigenvectors_loss(x_hat, eig_dec, criterion)
            loss += eigenstates_loss_weight * eig_loss
            total_eigenstates_loss += torch.mean(eig_loss).item()

        if diag_loss:
            diag_loss = diagonal_loss(x_hat, x, criterion, block_size=4)
            loss += diag_loss_weight * diag_loss
            total_diag_loss += torch.mean(diag_loss).item()


        loss.backward()
        vencoder_optimizer.step()
        decoder_optimizer.step()


    total_reconstruction_loss /= len(train_loader)
    total_edge_loss /= len(train_loader)
    total_eigenstates_loss /= len(train_loader)
    total_diag_loss /= len(train_loader)
    total_kl_loss /= len(train_loader)

    print(f'Reconstruction Loss: {total_reconstruction_loss}')
    print(f'KL Loss: {total_kl_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    if diag_loss:
        print(f'Diagonal Loss: {total_diag_loss}')
    print()

    return total_reconstruction_loss, total_kl_loss, total_edge_loss, total_eigenstates_loss, total_diag_loss