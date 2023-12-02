import typing as t
from tqdm import tqdm

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence


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
        if label is not None:
            y = y.to(device).squeeze()
            z = z[y == label]
            y = y[y == label]
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
