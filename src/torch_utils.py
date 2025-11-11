from __future__ import annotations

import torch
import numpy as np

from src.hamiltonian.hamiltonian import RepresentationMapping


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
    

def torch_majorana_polarization(
    H: torch.Tensor,
    axis: str = 'total',
    site: str = 'avg',
    num_mzm: int = 4,
    representation_mapping: RepresentationMapping = RepresentationMapping.default
):
    eigvals, eigvecs = torch.linalg.eig(H)    
    zm_indices = torch.topk(torch.abs(eigvals), num_mzm, largest=False, dim=-1).indices
    zm = torch.stack([eigvecs[i, :, zm_indices[i]] for i in range(eigvecs.shape[0])], dim=0)

    P_m = {}
    for i in range(zm.shape[1] // 4):
        zm_site_i = zm[:, 4*i:4*(i+1), :]
        P_m[i] = torch_majorana_polarization_site(zm_site_i, axis=axis, representation_mapping=representation_mapping)
        
    if site == 'avg':
        return torch.mean(torch.stack(list(P_m.values()), dim=-1), dim=-1)
    if site == 'sum':
        return torch.sum(torch.stack(list(P_m.values()), dim=-1), dim=-1)
    elif site == 'all':
        return P_m
    else:
        raise ValueError('site must be one of "avg", "all", or an integer')


def torch_majorana_polarization_site(zero_mode: torch.Tensor, axis: str = 'total', representation_mapping: RepresentationMapping = RepresentationMapping.default):
    pol = 0
    if axis == 'total':
        pol = 2*torch.mean(torch.abs(torch_majorana_polarization_product(zero_mode, representation_mapping)), dim=-1)
    if axis == 'x':
        pol = 2*torch.mean(torch.real(torch_majorana_polarization_product(zero_mode, representation_mapping)), dim=-1)
    if axis == 'y':
        pol = 2*torch.mean(torch.imag(torch_majorana_polarization_product(zero_mode, representation_mapping)), dim=-1)
    return pol


def torch_majorana_polarization_product(zero_mode: np.ndarray, representation_mapping: RepresentationMapping = RepresentationMapping.second_quantized_plus_minus_up_down):
    if representation_mapping == RepresentationMapping.majorana_plus_minus_up_down:
        return zero_mode[:, 1, :] * zero_mode[:, 3, :].conj() - zero_mode[:, 0, :] * zero_mode[:, 2, :].conj() # just guessing
    if representation_mapping == RepresentationMapping.second_quantized_polarization:
        '''Seems invalid for QDH (that's good actually)'''
        return zero_mode[:, 1, :] * zero_mode[:, 2, :].conj() + zero_mode[:, 0, :] * zero_mode[:, 3, :].conj()
    if representation_mapping == RepresentationMapping.second_quantized_plus_minus_up_down:
        return zero_mode[:, 1, :] * zero_mode[:, 3, :].conj() - zero_mode[:, 0, :] * zero_mode[:, 2, :].conj() # reversing conjugate makes the polarization flip signs - to discuss with Jarek
    else:
        raise ValueError('Invalid representation mapping')
