import typing as t
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.correlation_tools import cov_nearest
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence

from torch_utils import torch_total_polarization_loss

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


def kl_divergence_loss(q_dist):
    return kl_divergence(
        q_dist, Normal(torch.zeros_like(q_dist.mean), torch.ones_like(q_dist.stddev))
    )

def distribution_loss(x_hat: torch.Tensor, distribution: t.Tuple[torch.Tensor, torch.Tensor]):
    mean, stddev = distribution
    distance_from_distribution = torch.abs(x_hat - mean) - stddev
    loss = torch.maximum(distance_from_distribution, torch.zeros_like(distance_from_distribution))
    return torch.mean(loss)

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
    ev = torch.mul(eigvals, eigvec)
    xv = torch.matmul(x_hat_complex, eigvec)
    return criterion(torch.view_as_real(xv), torch.view_as_real(ev))


def determinant_loss(x_hat: torch.Tensor, eigvals: torch.Tensor, criterion: t.Callable):
    assert x_hat.shape[1] == 2, 'Wrong dimension of complex tensors'
    x_hat_complex = torch.complex(x_hat[:, 0, :, :], x_hat[:, 1, :, :])
    eigvals = torch.diag_embed(eigvals, dim1=-2, dim2=-1)
    x_hat_det = torch.logdet(x_hat_complex - eigvals)
    x_hat_det_real = torch.view_as_real(x_hat_det)
    return torch.mean(x_hat_det_real)


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
    return_ref_eigvals: bool = False,
    **kwargs: t.Dict[str, t.Any]
):
    H_rec = reconstruct_hamiltonian(matrix, encoder, decoder, **kwargs)
    auto_energies = np.linalg.eigvalsh(H_rec)

    if return_ref_eigvals:
        energies = np.linalg.eigvalsh(matrix)
        return auto_energies, energies
    else:
        return auto_energies


def reconstruct_hamiltonian(H: np.ndarray, encoder: nn.Module, decoder: nn.Module, device: torch.device = torch.device('cpu'), **kwargs: t.Dict[str, t.Any]):
    encoder.eval()
    decoder.eval()
    encoder.to(device)
    decoder.to(device)
    with torch.no_grad():
        H_torch = torch.from_numpy(H)
        H_torch = torch.stack((H_torch.real, H_torch.imag), dim= 0)
        H_torch = H_torch.unsqueeze(0).float().to(device)
        latent_vec = encoder(H_torch)
        if kwargs.get("decoder_eigvals", False):
            eigvals = torch.linalg.eigvalsh(torch.complex(H_torch[:, 0, :, :], H_torch[:, 1, :, :]))
            num_eigvals = kwargs.get("eigvals_num", H_torch.shape[-1])
            min_eigvals, min_eigvals_id = torch.topk(torch.abs(eigvals), num_eigvals, largest=False)
            # min_eigvals = torch.stack([eigvals[i, min_eigvals_id[i]] for i in range(len(eigvals))], dim=0) # uncomment for fixed eigvals
            if kwargs.get("shift_eigvals", False):
                min_eigvals = min_eigvals + torch.rand_like(min_eigvals) * kwargs.get("eigvals_shift", 1.)
            if kwargs.get("shift_latent", False):
                latent_vec = latent_vec + torch.rand_like(latent_vec) * kwargs.get("latent_shift", 1.)
            latent_vec = (latent_vec, min_eigvals.real)
        H_torch_rec = decoder(latent_vec)
        H_rec = torch.complex(H_torch_rec[:, 0, :, :], H_torch_rec[:, 1, :, :]).squeeze().cpu().numpy()
    return H_rec


def log_scale_loss(x: torch.Tensor, loss: torch.Tensor):
    scale_basis = torch.where(torch.abs(x) < 1.e-5, torch.ones_like(x), x)
    scale = torch.log10(torch.abs(scale_basis))
    factor = 10**(-scale)
    return factor * loss


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
    eigenvalues_loss: bool = False,
    eigenstates_loss: bool = False,
    diag_loss: bool = False,
    det_loss: bool = False,
    gt_eigvals = False,
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
    total_eigenvalues_loss = 0
    total_eigenstates_loss = 0
    total_diag_loss = 0
    total_det_loss = 0

    with torch.no_grad():
        for (x, _), eig_dec in tqdm(test_loader, "Testing autoencoder model"):
            x = x.to(device)
            if site_permutation:
                x = site_perm(x, encoder_model.N, encoder_model.block_size)
            z = encoder_model(x)
            if gt_eigvals:
                z = (z, eig_dec[0].to(device))
            x_hat = decoder_model(z)
            loss = criterion(x_hat, x)
            total_loss += loss.item()

            if edge_loss:
                e_loss = edge_diff(x_hat, x, criterion, edge_width=8)
                total_edge_loss += e_loss.item()

            if eigenvalues_loss:
                assert eig_dec is not None, "Incorrect eigen decomposition values"
                target_eigvals = eig_dec[0].to(device)
                if isinstance(z, tuple):
                    encoded_eigvals = z[1]
                else:
                    encoded_eigvals = torch.linalg.eigvals(z).real
                eigvals_loss = criterion(encoded_eigvals, target_eigvals)
                total_eigenvalues_loss += eigvals_loss.item()

            if det_loss:
                assert eig_dec is not None, "Incorrect eigen decomposition values"
                target_eigvals = eig_dec[0].to(device)
                det_loss = determinant_loss(x_hat, target_eigvals, criterion)
                total_det_loss += det_loss.item()

            if eigenstates_loss:
                assert eig_dec is not None, "Incorrect eigen decomposition values"
                eig_dec = eig_dec[0].to(device), eig_dec[1].to(device)
                eig_loss = eigenvectors_loss(x_hat, eig_dec, criterion)
                total_eigenstates_loss += eig_loss.item()

            if diag_loss:
                diag_loss = diagonal_loss(x_hat, x, criterion, block_size=4)
                total_diag_loss += diag_loss.item()

    total_loss /= len(test_loader)
    total_edge_loss /= len(test_loader)
    total_eigenvalues_loss /= len(test_loader)
    total_eigenstates_loss /= len(test_loader)
    total_diag_loss /= len(test_loader)
    total_det_loss /= len(test_loader)

    print(f'Loss: {total_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    print()

    return total_loss, total_edge_loss, total_eigenvalues_loss, total_eigenstates_loss, total_diag_loss, total_det_loss


def test_encoder_with_classifier(
    encoder_model: nn.Module,
    classifier_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
):
    class_criterion = nn.BCELoss()
    reg_criterion = nn.MSELoss(reduction='none')

    encoder_model.to(device)
    classifier_model.to(device)

    encoder_model.eval()
    classifier_model.eval()

    with torch.no_grad():
        total_class_loss = 0
        total_reg_loss = torch.zeros(classifier_model.output_dim - 1).to(device)
        conf_matrix = np.zeros((2, 2))

        for (x, y), _ in tqdm(test_loader, "Testing classifer model"):
            x = x.to(device)
            y = y.to(device)
            z = encoder_model(x)
            output = classifier_model(z)
            all_ids = torch.arange(y.shape[1])
            not_class_ids = all_ids[all_ids != classifier_model.classifier_output_idx]
            prediction = classifier_model(z)
            class_loss = class_criterion(prediction[:, classifier_model.classifier_output_idx], y[:, classifier_model.classifier_output_idx])
            reg_loss = reg_criterion(prediction[:, not_class_ids], y[:, not_class_ids])        
            total_class_loss += class_loss.item()
            total_reg_loss += torch.mean(reg_loss, dim=0)

            for i, j in zip(y[:, classifier_model.classifier_output_idx], prediction[:, classifier_model.classifier_output_idx]):
                conf_matrix[int(i.item()), round(j.item())] += 1

        total_class_loss /= len(test_loader)
        total_reg_loss /= len(test_loader)
        sensitivity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        specifity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
        bal_acc = 100.* (sensitivity + specifity) / 2

    print(f'Loss: {total_class_loss}, balanced accuracy: {bal_acc}')

    return total_class_loss, bal_acc, conf_matrix, total_reg_loss.cpu().tolist()


def test_generator_with_classifier(
    generator_model: nn.Module,
    # encoder_model: nn.Module,
    classifier_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
):
    criterion = nn.BCELoss()

    classifier_model.to(device)
    generator_model.to(device)
    # encoder_model.to(device)

    classifier_model.eval()
    generator_model.eval()
    # encoder_model.eval()

    total_loss = 0
    conf_matrix = np.zeros((2, 2))

    for (x, y), _ in tqdm(test_loader, 'Testing generator for classifier'):
        x = x.to(device)

        z = generator_model.get_noise(x.shape[0], device, noise_type='hybrid')
        x_hat = generator_model.noise_converter(z)
        # latent_prime = encoder_model(x_hat)
        y_hat = classifier_model(x_hat)
        desired_y = torch.ones_like(y_hat)

        loss = criterion(y_hat, desired_y)
        total_loss += loss.item()

        prediction = torch.round(y_hat)

        for i, j in zip(desired_y, prediction):
            conf_matrix[int(i), int(j)] += 1
    
    total_loss /= len(test_loader)
    specifity = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1])
    bal_acc = 100.* specifity

    print(f'Total loss: {total_loss}, balanced accuracy: {bal_acc}')

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
    eigenvalues_loss: bool = False,
    eigenvalues_loss_weight: float = .5,
    eigenstates_loss: bool = False,
    eigenstates_loss_weight: float = .5,
    diag_loss: bool = False,
    diag_loss_weight: float = .01,
    det_loss: bool = False,
    det_loss_weight: float = .01,
    log_scaled_loss: bool = False,
    gt_eigvals = False,
):
    if site_permutation and edge_loss:
        raise NotImplementedError("Combining edge loss with site permutation is not implemented")

    if log_scaled_loss:
        assert not diag_loss, "Diagonal loss is not implemented for log scaled loss"
        criterion = nn.MSELoss(reduction='none')
    else:
        criterion = nn.MSELoss()

    encoder_model.to(device)
    decoder_model.to(device)

    encoder_model.train()
    decoder_model.train()

    total_loss = 0
    total_edge_loss = 0
    total_eigenvalues_loss = 0
    total_eigenstates_loss = 0
    total_diag_loss = 0
    total_det_loss = 0

    print(f'Epoch: {epoch}')
    for (x, _), eig_dec in tqdm(train_loader, 'Training autoencoder model'):
        x = x.to(device)
        if site_permutation:
            x = site_perm(x, encoder_model.N, encoder_model.block_size)
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        z = encoder_model(x)
        if gt_eigvals:
            z = (z, eig_dec[0].to(device))
        x_hat = decoder_model(z) 
        loss = criterion(x_hat, x)
        total_loss += torch.mean(loss).item()

        if edge_loss:
            e_loss = edge_diff(x_hat, x, criterion, edge_width=8)
            loss += edge_loss_weight * e_loss
            total_edge_loss += torch.mean(e_loss).item()

        if eigenvalues_loss:
            assert eig_dec is not None, "Incorrect eigen decomposition values"
            if isinstance(z, tuple):
                encoded_eigvals = z[1]
            else:
                encoded_eigvals = torch.linalg.eigvals(x_hat)
            target_eigvals = eig_dec[0].to(device)
            eig_loss = criterion(encoded_eigvals, target_eigvals)
            loss += eigenvalues_loss_weight * eig_loss
            total_eigenvalues_loss += torch.mean(eig_loss).item()

        if det_loss:
            assert eig_dec is not None, "Incorrect eigen decomposition values"
            target_eigvals = eig_dec[0].to(device)
            det_loss = determinant_loss(x_hat, target_eigvals, criterion)
            loss += det_loss_weight * det_loss
            total_det_loss += torch.mean(det_loss).item()

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

        if log_scaled_loss:
            loss = log_scale_loss(x, loss)    
            loss = torch.mean(loss)
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

    
    total_loss /= len(train_loader)
    total_edge_loss /= len(train_loader)
    total_eigenvalues_loss /= len(train_loader)
    total_eigenstates_loss /= len(train_loader)
    total_diag_loss /= len(train_loader)
    total_det_loss /= len(train_loader)

    print(f'Loss: {total_loss}')
    if edge_loss:
        print(f'Edge Loss: {total_edge_loss}')
    if eigenstates_loss:
        print(f'Eigenstates Loss: {total_eigenstates_loss}')
    print()

    return total_loss, total_edge_loss, total_eigenvalues_loss, total_eigenstates_loss, total_diag_loss, total_det_loss


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

        k1_loss = kl_divergence_loss(freq_dist).mean() + kl_divergence_loss(block_dist).mean()
        total_kl_loss += k1_loss.item()
        loss += .01 * k1_loss

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


def train_encoder_with_classifier(
    encoder_model: nn.Module,
    decoder_model: nn.Module,
    classifier_model: nn.Module,
    train_loader: torch.utils.data.DataLoader, 
    epoch: int,
    device: torch.device, 
    encoder_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    classifier_optimizer: torch.optim.Optimizer,
    gt_eigvals: bool = False,
    class_loss_weight: float = 0.01
):

    class_criterion = nn.BCELoss()
    ae_criterion = nn.MSELoss()

    encoder_model.to(device)
    classifier_model.to(device)
    decoder_model.to(device)

    encoder_model.eval()
    classifier_model.train()

    total_loss_class = 0
    total_loss_ae = 0

    print(f'Epoch: {epoch}')
    for (x, y), eig_dec in tqdm(train_loader, 'Training classifier model'):
        x = x.to(device)
        y = y.to(device)
        classifier_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        z = encoder_model(x)

        # reduce the examples so that the number of examples in each class is the same
        # y_sq = y.squeeze()
        # z_reduced = torch.cat((z[y_sq == 0][:len(z[y_sq == 1])], z[y_sq == 1]))
        # y_reduced = torch.cat((y[y_sq == 0][:len(z[y_sq == 1])], y[y_sq == 1]))
        all_ids = torch.arange(y.shape[1])
        not_class_ids = all_ids[all_ids != classifier_model.classifier_output_idx]
        prediction = classifier_model(z)
        loss_class = class_criterion(prediction[:, classifier_model.classifier_output_idx], y[:, classifier_model.classifier_output_idx]) +\
                    ae_criterion(prediction[:, not_class_ids], y[:, not_class_ids])
        total_loss_class += loss_class.item()
        # if len(z_reduced) > 0:
        #     total_loss_class += loss_class.item()

        if gt_eigvals:
            z = (z, eig_dec[0].to(device))
        x_hat = decoder_model(z) 
        loss_ae = ae_criterion(x_hat, x)

        total_loss_ae += loss_ae.item()

        loss =class_loss_weight*loss_class + loss_ae
        loss.backward()
        classifier_optimizer.step()
        encoder_optimizer.step()
        decoder_optimizer.step()
    
    total_loss_ae /= len(train_loader)
    total_loss_class /= len(train_loader)

    print(f'Classification loss: {total_loss_class}\n')
    print(f'Autoencoder loss: {total_loss_ae}\n')

    return total_loss_class, total_loss_ae


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


def calculate_classifier_distance(
    x1: np.array,
    x2: np.array,
    model: nn.Module,
):
    x1 = torch.from_numpy(x1)
    x2 = torch.from_numpy(x2)
    x1 = x1.unsqueeze(0).float()
    x2 = x2.unsqueeze(0).float()
    z1 = model(x1)
    z2 = model(x2)
    return torch.linalg.norm(z1 - z2).item()

def calculate_latent_space_distribution(
    encoder: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    label: t.Optional[int] = None,
):
    encoder.to(device)
    encoder.eval()

    # get latent space shape
    (x, _), _ = next(iter(data_loader))
    x = x.to(device)
    z: torch.Tensor = encoder(x)

    # calculate latent space distribution (mean and std)
    population = []
    for (x, y), _ in tqdm(data_loader, 'Calculating latent space mean'):
        x = x.to(device)
        if label is not None:
            y = y.to(device).squeeze()
            x = x[y == label]
        if x.shape[0] > 0:
            z = encoder(x)
            population.append(z)
    population = torch.cat(population, dim=0)
    # space_dim = population.shape[1]
    # covariance_matrix_freq = torch.cov(population.T[:space_dim//2])
    # covariance_matrix_block = torch.cov(population.T[space_dim//2:])
    covariance_matrix = torch.cov(population.T)
    mean = torch.mean(population, dim=0)
    std = torch.std(population, dim=0)

    return mean, std, covariance_matrix


def calculate_pca(
    encoder_model: nn.Module,
    test_loader: torch.utils.data.DataLoader, 
    device: torch.device, 
    label: t.Optional[int] = None,
    n_components: int = 2,
):
    encoder_model.to(device)
    encoder_model.eval()

    z_list = []
    y_list = []

    for (x, y), _ in tqdm(test_loader, "Testing dim-red"):
        x = x.to(device)
        z = encoder_model(x).detach().cpu().numpy()
        # if label is not None:
        #     y = y.to(device).squeeze()
        #     z = z[y == label]
        #     y = y[y == label]
        z_list.append(z)
        y_list.append(y.detach().cpu().numpy())

    z = np.concatenate(z_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    if label is not None:
        z2 = z[y.squeeze() == label]

    pca = PCA(n_components=n_components)
    pca.fit(z)
    pca_z2 = pca.transform(z2)
    return pca, pca_z2, z2


def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0
    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def is_pos_semidef(x):
    return np.all(np.linalg.eigvals(x) >= 0)


def generate_sample_from_mean_and_covariance(mean: torch.Tensor, covariance_matrix: torch.Tensor, batch_size: int = 1):
    mvn = MultivariateNormal(mean, covariance_matrix)
    return mvn.sample((batch_size,))
