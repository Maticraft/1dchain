
from __future__ import annotations
from copy import deepcopy
import typing as t

import numpy as np
import torch

from src.hamiltonian.utils import Hamiltonian, calculate_gap, calculate_mzm_main_bands_gap, count_mzm_states, majorana_polarization, majoranization
from src.hamiltonian.hamiltonian_torch_handlers import HamiltonianConverter, HamiltonianParams


class TorchHamiltonian(Hamiltonian):
    def __init__(self, hamiltonian: torch.Tensor, hamiltonian_params: t.Optional[HamiltonianParams] = None):
        self.hamiltonian = hamiltonian.detach().cpu()
        self.threshold = 0.05
        self.parameters = hamiltonian_params
        if self.parameters is not None:
            self.converter = HamiltonianConverter(self.parameters)
    
    @property
    def param_map(self) -> t.Optional[t.Dict[str, t.Dict[str, torch.Tensor]]]:
        if not self.parameters:
            raise NotImplementedError('TorchHamiltonian does not support getting parameters, when no HamiltonianParams are provided')
        hamiltonian_2channels = torch.stack((self.hamiltonian.real, self.hamiltonian.imag), dim=0)
        return self.converter.from_matrix_to_params(hamiltonian_2channels.unsqueeze(0))

    def set_parameter(self, parameter_name: str, value: t.Any):
        new_param_map = deepcopy(self.param_map)
        param_keys = self.get_main_param_keys(parameter_name)
        if len(param_keys) == 0:
            raise ValueError(f'Parameter {parameter_name} not found in Hamiltonian parameters')

        for key in param_keys:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if not isinstance(value, torch.Tensor):
                value = torch.full_like(new_param_map[key][parameter_name], value)

            if 'real' in key:
                new_param_map[key][parameter_name] = value.real.unsqueeze(0)
            elif 'imag' in key:
                new_param_map[key][parameter_name] = value.imag.unsqueeze(0)
        
        new_hamiltonian = self.converter.from_params_to_matrix(new_param_map).squeeze(0)
        self.hamiltonian = new_hamiltonian[0] + 1j*new_hamiltonian[1]

    def get_parameter(self, parameter_name: str, reduction: t.Optional[str] = None, exact_index: t.Optional[int] = None) -> t.Any:
        param_keys = self.get_main_param_keys(parameter_name)

        if len(param_keys) == 0:
            raise ValueError(f'Parameter {parameter_name} not found in Hamiltonian parameters')
        
        value = 0.
        for key in param_keys:
            param_value = self.param_map[key][parameter_name]
            if 'real' in key:
                value = value + param_value
            elif 'imag' in key:
                value = value + 1.j*param_value
        
        if exact_index is None:
            match reduction:
                case 'mean':
                    return value.mean().item()
                case None:
                    return value.squeeze(0).numpy()
        
        return value[exact_index].item()
    
    def get_main_param_keys(self, parameter_name: str) -> t.List[str]:
        main_keys = []
        for key, key_dict in self.param_map.items():
            if parameter_name in key_dict:
                main_keys.append(key)
        return main_keys

    @classmethod
    def from_2channel_tensor(cls, hamiltonian: torch.Tensor) -> TorchHamiltonian:
        hamiltonian = hamiltonian[0] + 1j*hamiltonian[1]
        return cls(hamiltonian)
    
    @classmethod
    def from_2channel_tensor_with_params(cls, hamiltonian: torch.Tensor, hamiltonian_params: HamiltonianParams) -> TorchHamiltonian:
        hamiltonian = hamiltonian[0] + 1j*hamiltonian[1]
        return cls(hamiltonian, hamiltonian_params)

    def get_hamiltonian_matrix(self) -> np.ndarray:
        return self.hamiltonian.numpy()

    def get_label(self) -> str:
        h = self.hamiltonian.numpy()
    
        mp = majorana_polarization(h, threshold=self.threshold, axis='y', site='all')
        values = list(mp.values())
        mp_y_sum_left = sum(values[:len(values)//2])
        mp_y_sum_right = sum(values[len(values)//2:])
        mp_tot = majorana_polarization(h, threshold=self.threshold, axis='total', site='all')
        values_tot = list(mp_tot.values())
        mp_tot_sum_left = sum(values_tot[:len(values_tot)//2])
        mp_tot_sum_right = sum(values_tot[len(values_tot)//2:])
        band_gap = calculate_gap(h)
        num_mzm = count_mzm_states(h, threshold=self.threshold)
        if num_mzm > 0:
            mzm_gap = calculate_mzm_main_bands_gap(h, mzm_threshold=None, num_majoranas=2)
        else:
            mzm_gap = 0

        m_projection = majoranization(h, self.parameters.no_dots) 
        return f"{mp_tot_sum_left}, {mp_tot_sum_right}, {mp_y_sum_left}, {mp_y_sum_right}, {num_mzm}, {band_gap}, {mzm_gap}, {m_projection}"

    @property
    def num_sites(self) -> int:
        return self.hamiltonian.shape[-1] // 4
