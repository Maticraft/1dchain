import typing as t

import torch
import numpy as np

from src.data_utils import Hamiltonian
from src.hamiltonian.utils import count_mzm_states, majorana_polarization


class TorchHamiltonian(Hamiltonian):
    def __init__(self, hamiltonian: torch.Tensor):
        self.hamiltonian = hamiltonian.detach().cpu()    

    def from_2channel_tensor(hamiltonian: torch.Tensor) -> 'TorchHamiltonian':
        hamiltonian = hamiltonian[0] + 1j*hamiltonian[1]
        return TorchHamiltonian(hamiltonian)

    def get_hamiltonian(self) -> np.ndarray:
        return self.hamiltonian.numpy()

    def get_label(self) -> str:
        h = self.hamiltonian.numpy()
        mp_tot = majorana_polarization(h, threshold=0.05, axis='total', site='all')
        values_tot = list(mp_tot.values())
        mp_tot_sum_left = sum(values_tot[:len(values_tot)//2])
        mp_tot_sum_right = sum(values_tot[len(values_tot)//2:])
        return f"{mp_tot_sum_left}, {mp_tot_sum_right}, {count_mzm_states(self.H, threshold=0.05)}"


def torch_total_polarization_loss(x_hat: torch.Tensor) -> torch.Tensor:
    h = x_hat[:, 0] + 1j*x_hat[:, 1]
    mp_tot, eigvals_loss = torch_extended_majorana_polarization(h, axis='total', site='all', num_mzm=4)
    values_tot = torch.stack(list(mp_tot.values()), dim=-1)
    mp_tot_sum_left = torch.sum(values_tot[:, :len(values_tot)//2], dim=-1)
    mp_tot_sum_right = torch.sum(values_tot[:, len(values_tot)//2:], dim=-1)
    eigvals_loss = torch.sum(eigvals_loss, dim=-1)
    return torch.mean(torch.abs(mp_tot_sum_left - 0.5) + torch.abs(mp_tot_sum_right - 0.5) + eigvals_loss)


def torch_extended_majorana_polarization(
    H: torch.Tensor,
    axis: str = 'total',
    site: str = 'avg',
    num_mzm: int = 4
):
    eigvals, eigvecs = torch.linalg.eig(H)    
    indices = torch.argsort(torch.abs(eigvals), dim=-1)
    num_eigvals = eigvals.shape[-1]

    majoranas_ids = indices[:, :num_mzm]
    not_majoranas_ids = indices[:, num_mzm:2*num_mzm] # taking the same number of non-MZMs as MZMs

    majoranas_loss_weight = num_eigvals / num_mzm
    not_majoranas_loss_weight = num_eigvals / num_mzm # (num_eigvals - num_mzm)

    eigvals_loss = torch.zeros_like(eigvals, dtype=torch.float32)
    zm = torch.zeros_like(eigvecs) 
    for i in range(eigvals.shape[0]):
        eigvals_loss[i, majoranas_ids[i, :]] = majoranas_loss_weight * torch.abs(eigvals[i, majoranas_ids[i, :]]) # eigenvalues should be 0 for MZMs
        eigvals_loss[i, not_majoranas_ids[i, :]] = not_majoranas_loss_weight / (2 + torch.abs(eigvals[i, not_majoranas_ids[i, :]])) # eigenvalues should larger than 0 for non-MZMs
        zm[i, :, majoranas_ids[i, :]] = eigvecs[i, :, majoranas_ids[i, :]]

    P_m = {}
    for i in range(zm.shape[1] // 4):
        zm_site_i = zm[:, 4*i:4*(i+1), :]
        P_m[i] = torch_majorana_polarization_site(zm_site_i, num_mzm, axis=axis)
        
    if site == 'avg':
        return torch.mean(torch.stack(list(P_m.values()), dim=-1), dim=-1), torch.mean(eigvals_loss, dim=-1)
    if site == 'sum':
        return torch.sum(torch.stack(list(P_m.values()), dim=-1), dim=-1), torch.sum(eigvals_loss, dim=-1)
    elif site == 'all':
        return P_m, eigvals_loss
    else:
        raise ValueError('site must be one of "avg", "all", or an integer')


def torch_majorana_polarization_site(zero_mode: torch.Tensor, num_eigs: torch.Tensor, axis: str = 'total', eps: float = 1e-10):
    pol = 0
    if axis == 'total':
        pol = 2*torch.sum(torch.abs(zero_mode[:, 1, :] * zero_mode[:, 2, :].conj() + zero_mode[:, 0, :] * zero_mode[:, 3, :].conj()), dim=-1)
    if axis == 'x':
        pol = 2*torch.sum(torch.real(zero_mode[:, 1, :] * zero_mode[:, 2, :].conj() + zero_mode[:, 0, :] * zero_mode[:, 3, :].conj()), dim=-1)
    if axis == 'y':
        pol = 2*torch.sum(torch.imag(zero_mode[:, 1, :] * zero_mode[:, 2, :].conj() + zero_mode[:, 0, :] * zero_mode[:, 3, :].conj()), dim=-1)
    return pol / (num_eigs + eps)