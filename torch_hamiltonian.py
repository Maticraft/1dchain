import torch
import numpy as np

from data_utils import Hamiltonian
from majorana_utils import count_mzm_states, majorana_polarization


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
