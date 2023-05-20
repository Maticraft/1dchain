import typing as t
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn


def diagonal_loss(x_hat: torch.Tensor, x: torch.Tensor, criterion: t.Callable, block_size: int = 4):
    x_hat_strip = get_diagonal_strip(x_hat, block_size)
    x_strip = get_diagonal_strip(x, block_size)
    diff = criterion(x_hat_strip, x_strip)
    return diff


def get_diagonal_strip(x: torch.Tensor, block_size: int):
    strip = torch.zeros((x.shape[0], x.shape[1], block_size, x.shape[3])).to(x.device)
    N = x.shape[2] // block_size
    for i in range(N):
        idx0 =  i*block_size
        idx1 = (i+1)*block_size
        strip[:, :, :, idx0: idx1] = x[:, :, idx0: idx1, idx0: idx1]
    return strip


def edge_diff(x_hat: torch.Tensor, x: torch.Tensor, criterion: t.Callable, edge_width: int = 4):
    x_hat_edges = get_edges(x_hat, edge_width)
    x_edges = get_edges(x, edge_width)
    diff = criterion(x_hat_edges[x_hat_edges != 0.], x_edges[x_hat_edges != 0.])
    return diff


def eigenvectors_loss(x_hat: torch.Tensor, eig_dec: t.Tuple[torch.Tensor, torch.Tensor], criterion: t.Callable, zmt: float = 1e5):
    assert x_hat.shape[1] == 2, 'Wrong dimension of complex tensors'
    x_hat_complex = torch.complex(x_hat[:, 0, :, :], x_hat[:, 1, :, :])
    eigvals, eigvec = eig_dec
    eigvals = eigvals.unsqueeze(dim=1).expand(-1, eigvec.shape[1], -1)
    eigvec[torch.abs(eigvals) < (1 / zmt)] *= 1000
    ev = torch.mul(eigvals, eigvec)
    xv = torch.matmul(x_hat_complex, eigvec)
    return criterion(torch.view_as_real(xv), torch.view_as_real(ev))


def get_edges(x: torch.Tensor, edge_width: int):
    edges = torch.zeros_like(x)
    edges[:, :, :edge_width, :] = x[:, :, :edge_width, :]
    edges[:, :, -edge_width:, :] = x[:, :, -edge_width:, :]
    edges[:, :, :, :edge_width] = x[:, :, :, :edge_width]
    edges[:, :, :, -edge_width:] = x[:, :, :, -edge_width:]
    return edges


def get_eigvals(    
    matrix: np.ndarray,
    encoder: nn.Module,
    decoder: nn.Module,
    device: t.Optional[torch.device] = None,
    return_ref_eigvals: bool = False,
):

    H_rec = reconstruct_hamiltonian(matrix, encoder, decoder, device)
    auto_energies = np.linalg.eigvalsh(H_rec)

    if return_ref_eigvals:
        energies = np.linalg.eigvalsh(matrix)
        return auto_energies, energies
    else:
        return auto_energies


def reconstruct_hamiltonian(H: np.ndarray, encoder: nn.Module, decoder: nn.Module, device: torch.device = torch.device('cpu')):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        H_torch = torch.from_numpy(H)
        H_torch = torch.stack((H_torch.real, H_torch.imag), dim= 0)
        H_torch = H_torch.unsqueeze(0).float().to(device)

        latent_vec = encoder(H_torch)
        H_torch_rec = decoder(latent_vec)
        H_rec = torch.complex(H_torch_rec[:, 0, :, :], H_torch_rec[:, 1, :, :]).squeeze().cpu().numpy()
    return H_rec


def log_scale_loss(x: torch.Tensor, loss: torch.Tensor):
    scale_basis = torch.where(torch.abs(x) < 1.e-5, torch.ones_like(x), x)
    scale = torch.log10(torch.abs(scale_basis))
    factor = 10**(-scale)
    return factor * loss


def mask_loss_fn(x: torch.Tensor, params: t.Dict[str, t.Any]):
    criterion = nn.BCELoss()
    string_width = params['M']
    specific_positions = params['potential_positions']
    left_edge = params['potential_before']
    right_edge = params['potential_after']

    mask = torch.zeros_like(x)
    mask[:, :, :, :left_edge*string_width] = 1.
    mask[:, :, :, right_edge*string_width:] = 1.
    for pos in specific_positions:
        mask[:, :, :, pos['i']*string_width + pos['j']] = 1.

    loss = criterion(x, mask)
    return loss


def site_perm(x: torch.Tensor, N: int, block_size: int):
    permutation = torch.randperm(N)
    permuted_indices = torch.cat([torch.arange(i*block_size, (i+1)*block_size) for i in permutation], dim=0)
    x_permuted = x[:, :, permuted_indices, :][:, :, :, permuted_indices]
    return x_permuted


def test_autoencoder(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    site_permutation: bool = False,
    edge_loss: bool = False,
    eigenstates_loss: bool = False,
    diag_loss: bool = False,
    mask_loss: bool = False,
):
    if site_permutation and edge_loss:
        raise NotImplementedError("Combining edge loss with site permutation is not implemented")

    criterion = nn.MSELoss()

    encoder_model.to(device)
    decoder_model.to(device)

    encoder_model.eval()
    decoder_model.eval()

    total_loss = 0
    total_edge_loss = 0
    total_eigenstates_loss = 0
    total_diag_loss = 0
    total_mask_loss = 0

    for data in tqdm(test_loader, "Testing autoencoder model"):
        if mask_loss:
            (x, _), eig_dec, params = data
        else:
            (x, _), eig_dec = data

        x = x.to(device)
        if site_permutation:
            x = site_perm(x, encoder_model.N, encoder_model.block_size)
        z = encoder_model(x)
        
        if mask_loss:
            x_hat, mask_tensor = decoder_model(z, return_mask=mask_loss)
        else:
            x_hat = decoder_model(z)

        loss = criterion(x_hat, x)
        total_loss += loss.item()

        if edge_loss:
            e_loss = edge_diff(x_hat, x, criterion, edge_width=8)
            total_edge_loss += e_loss.item()

        if eigenstates_loss:
            assert eig_dec is not None, "Incorrect eigen decomposition values"
            eig_dec = eig_dec[0].to(device), eig_dec[1].to(device)
            eig_loss = eigenvectors_loss(x_hat, eig_dec, criterion)
            total_eigenstates_loss += eig_loss.item()

        if diag_loss:
            d_loss = diagonal_loss(x_hat, x, criterion, block_size=4)
            total_diag_loss += d_loss.item()

        if mask_loss:
            m_loss = mask_loss_fn(mask_tensor, params)
            total_mask_loss += m_loss.item()

    total_loss /= len(test_loader)
    total_edge_loss /= len(test_loader)
    total_eigenstates_loss /= len(test_loader)
    total_diag_loss /= len(test_loader)
    total_mask_loss /= len(test_loader)

    print(f'Loss: {total_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    if diag_loss:
        print(f'Diagonal Loss: {total_diag_loss}')
    if mask_loss:
        print(f'Mask Loss: {total_mask_loss}')
    print()

    return total_loss, total_edge_loss, total_eigenstates_loss, total_diag_loss, total_mask_loss


def test_classifier(
    encoder_model: nn.Module,
    classifier_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
):
    criterion = nn.BCELoss()

    encoder_model.to(device)
    classifier_model.to(device)

    encoder_model.eval()
    classifier_model.eval()

    total_loss = 0
    conf_matrix = np.zeros((2, 2))

    for (x, y), _ in tqdm(test_loader, "Testing classifer model"):
        x = x.to(device)
        z = encoder_model(x)
        output = classifier_model(z)
        loss = criterion(prediction, y)
        total_loss += loss.item()

        prediction = torch.round(output)              

        for i, j in zip(y, prediction):
            conf_matrix[int(i), int(j)] += 1

    total_loss /= len(test_loader)
    sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
    specifity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    bal_acc = 100.* (sensitivity + specifity) / 2

    print(f'Loss: {total_loss}, balanced accuracy: {bal_acc}')

    return total_loss, bal_acc, conf_matrix


def train_autoencoder(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    train_loader: torch.utils.data.DataLoader, 
    epoch: int,
    device: torch.device, 
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    site_permutation: bool = False,
    edge_loss: bool = False,
    edge_loss_weight: float = .5,
    eigenstates_loss: bool = False,
    eigenstates_loss_weight: float = .5,
    diag_loss: bool = False,
    diag_loss_weight: float = .01,
    log_scaled_loss: bool = False,
    mask_loss: bool = False,
):
    if site_permutation and edge_loss:
        raise NotImplementedError("Combining edge loss with site permutation is not implemented")

    if log_scaled_loss:
        assert not diag_loss, "Diagonal loss is not implemented for log scaled loss"
        assert not mask_loss, "Mask loss is not implemented for log scaled loss"
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.MSELoss()

    encoder_model.to(device)
    decoder_model.to(device)

    encoder_model.train()
    decoder_model.train()

    total_loss = 0
    total_edge_loss = 0
    total_eigenstates_loss = 0
    total_diag_loss = 0
    total_mask_loss = 0

    print(f'Epoch: {epoch}')
    for data in tqdm(train_loader, 'Training model'):
        if mask_loss:
            (x, _), eig_dec, params = data
        else:
            (x, _), eig_dec = data

        x = x.to(device)
        if site_permutation:
            x = site_perm(x, encoder_model.N, encoder_model.block_size)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        z = encoder_model(x)
        if mask_loss:
            x_hat, mask_tensor = decoder_model(z, return_mask=mask_loss)
        else:
            x_hat = decoder_model(z)
        loss = criterion(x_hat, x)
        total_loss += torch.mean(loss).item()

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
            d_loss = diagonal_loss(x_hat, x, criterion, block_size=4)
            loss += diag_loss_weight * d_loss
            total_diag_loss += torch.mean(d_loss).item()

        if mask_loss:
            m_loss = mask_loss_fn(mask_tensor, params)
            loss += m_loss
            total_mask_loss += torch.mean(m_loss).item()

        if log_scaled_loss:
            loss = log_scale_loss(x, loss)    
            loss = torch.mean(loss)

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    
    total_loss /= len(train_loader)
    total_edge_loss /= len(train_loader)
    total_eigenstates_loss /= len(train_loader)
    total_diag_loss /= len(train_loader)
    total_mask_loss /= len(train_loader)

    print(f'Loss: {total_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    if diag_loss:
        print(f'Diagonal Loss: {total_diag_loss}')
    if mask_loss:
        print(f'Mask Loss: {total_mask_loss}')
    print()

    return total_loss, total_edge_loss, total_eigenstates_loss, total_diag_loss, total_mask_loss


def train_classifier(
    encoder_model: nn.Module,
    classifier_model: nn.Module,
    train_loader: torch.utils.data.DataLoader, 
    epoch: int,
    device: torch.device, 
    optimizer: torch.optim.Optimizer,
):

    criterion = nn.BCELoss()

    encoder_model.to(device)
    classifier_model.to(device)

    encoder_model.eval()
    classifier_model.train()

    total_loss = 0

    print(f'Epoch: {epoch}')
    for (x, y), _ in tqdm(train_loader, 'Training model'):
        x = x.to(device)
        optimizer.zero_grad()

        z = encoder_model(x).detach()

        prediction = classifier_model(z)
        loss = criterion(prediction, y)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    total_loss /= len(train_loader)

    print(f'Loss: {total_loss}\n')

    return total_loss
