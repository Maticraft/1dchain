import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from simpleML_training import MODEL_NAME, MODEL_SAVE_DIR
from data_utils import Hamiltonian, generate_data
from majorana_utils import count_mzm_states, majorana_polarization, plot_eigvals, plot_eigvec, plot_majorana_polarization

DEFAULT_PARAMS = {'N': 70, 'M': 2, 'delta': 0.3, 'mu': 0.9, 'J': 1., 'delta_q': np.pi, 't': 1}


class SpinLadder(Hamiltonian):
    def __init__(self, N, M, mu = 0.9, delta = 0.3, J = 1, q = np.pi/2, delta_q = np.pi, t = 1, S = 1, theta = np.pi/2, B = 0.):
        self.block_size = 4
        self.M = M
        self.N = N
        self.t = t
        self.mu = mu
        self.delta = delta
        self.delta_q = delta_q
        self.q = q
        self.J = J
        self.S = S
        self.theta = theta
        self.B = B
        self.H = self._construct_open_boundary_hamiltonian(self.N, self.M)

    def _construct_open_boundary_hamiltonian(self, N, M):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        H = np.zeros((self.block_size*N*M, self.block_size*N*M), dtype=np.complex128)
        for i in range(N):
            for j in range(M):

                H[self._idx(i, j) : self._idx(i, j) + self.block_size, self._idx(i, j) : self._idx(i, j) + self.block_size] = self._spin_block_i_cdagger_c(self.mu, i, j, self.theta, self.delta_q, self.q, self.J*self.S, self.B)

                k = self.block_size // 2
                H[self._idx(i, j) : self._idx(i, j) + k, self._idx(i, j) + k : self._idx(i, j) + self.block_size] += self._delta_block_i_cdagger_cdagger(self.delta)
                H[self._idx(i, j) + k : self._idx(i, j) + self.block_size, self._idx(i, j) : self._idx(i, j) + k] += np.conjugate(self._delta_block_i_cdagger_cdagger(self.delta)).T

                if i < N-1:
                    H[self._idx(i, j) : self._idx(i, j) + self.block_size, self._idx(i + 1, j) : self._idx(i + 1, j) + self.block_size] = np.kron(sigma_z, self._interaction_block_ij_cdagger_c(self.t))
                    H[self._idx(i + 1, j) : self._idx(i + 1, j) + self.block_size, self._idx(i, j) : self._idx(i, j) + self.block_size] = np.conjugate(np.kron(sigma_z, self._interaction_block_ij_cdagger_c(self.t))).T

                if j < M-1:  
                    H[self._idx(i, j) : self._idx(i, j) + self.block_size, self._idx(i, j + 1) : self._idx(i, j + 1) + self.block_size] = np.kron(sigma_z, self._interaction_block_ij_cdagger_c(self.t))
                    H[self._idx(i, j + 1) : self._idx(i, j + 1) + self.block_size, self._idx(i, j) : self._idx(i, j) + self.block_size] = np.conjugate(np.kron(sigma_z, self._interaction_block_ij_cdagger_c(self.t))).T

        return H

    def _idx(self, i, j):
        return (i*self.M + j)*self.block_size

    def _spin_block_i_cdagger_c(self, mu, i, j, theta, delta_q, q, S, B):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        mu_block = -mu*np.eye(2, dtype=np.complex128)
        
        block = np.kron(sigma_z, mu_block)
        
        phi = i*q + j*delta_q
        spin_block = self._get_spin_matrix(theta, phi, S, B)

        block[0:2, 0:2] += spin_block
        block[2:4, 2:4] += -spin_block.T

        return block


    def _delta_block_i_cdagger_cdagger(self, delta):
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        delta_block = 1j*sigma_y*delta
        return delta_block

    
    def _interaction_block_ij_cdagger_c(self, t):
        interaction_block = -t*np.eye(2, dtype=np.complex128)
        return interaction_block


    def _get_spin_matrix(self, theta, phi, S, B=1.e-7):
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        spin_matrix = S*(np.sin(theta)*np.cos(phi)*pauli_x + np.sin(theta)*np.sin(phi)*pauli_y + np.cos(theta)*pauli_z)/2

        # Add magnetic field
        spin_matrix += B*pauli_z

        return spin_matrix


    def get_hamiltonian(self):
        return self.H

    
    def get_label(self):
        mp = majorana_polarization(self.H, threshold=1.e-5, axis='y', site='all')
        values = list(mp.values())
        mp_y_sum_left = sum(values[:len(values)//2])
        mp_y_sum_right = sum(values[len(values)//2:])
        mp_tot = majorana_polarization(self.H, threshold=1.e-5, axis='total', site='all')
        values_tot = list(mp_tot.values())
        mp_tot_sum_left = sum(values_tot[:len(values_tot)//2])
        mp_tot_sum_right = sum(values_tot[len(values_tot)//2:])
        return f"{mp_tot_sum_left}, {mp_tot_sum_right}, {mp_y_sum_left}, {mp_y_sum_right}, {count_mzm_states(self.H, threshold=1.e-5)}, {self.N}, {self.M}, {self.delta}, {self.q}, {self.mu}, {self.J}, {self.delta_q}, {self.t}, {self.theta}"



def generate_param_data(N, M, N_samples, flename):
    data = pd.DataFrame(columns=['N', 'M', 'delta', 'q', 'mu', 'J', 'delta_q', 't', 'theta', 'mp_tot_l', 'mp_tor_r', 'mp_y_l', 'mp_y_r', 'num_zm'])

    deltas = np.random.normal(1.8, 1, size= N_samples)
    qs = np.concatenate((np.random.normal(1.8, 0.5, size= N_samples // 2), np.random.normal(4.3, 0.5, size= N_samples // 2)))
    mus = np.random.normal(1.8, 1, size= N_samples)
    Js = np.random.normal(1.8, 1, size= N_samples)
    delta_qs = np.concatenate((np.random.normal(0.2, 0.5, size= N_samples // 2), np.random.normal(5.9, 0.5, size= N_samples // 2)))
    ts = np.random.normal(1, 0.5, size= N_samples)
    theta = np.pi / 2

    for i in tqdm(range(N_samples), desc='Generating data'): 
        ladder = SpinLadder(N, M, mus[i], deltas[i], Js[i], qs[i], delta_qs[i], ts[i], theta = theta)
        mp_tot_l, mp_tor_r, mp_y_l, mp_y_r, num_zm = ladder.get_label().split(', ')
        data.loc[i] = [N, M, deltas[i], qs[i], mus[i], Js[i], delta_qs[i], ts[i], theta, mp_tot_l, mp_tor_r, mp_y_l, mp_y_r, num_zm]

    data.to_csv(flename, index=False)


def generate_params(N, M, N_samples, N_qs):
    deltas = np.random.normal(1.8, 1, size= N_samples)
    qs = np.arange(0, 2*np.pi, 2*np.pi / N_qs)
    delta_qs = np.arange(0, 2*np.pi, 2*np.pi / N_qs)
    q_ids = np.random.choice(N_qs, size= N_samples)
    mus = np.random.normal(1.8, 1, size= N_samples)
    Js = np.random.normal(1.8, 1, size= N_samples)
    ts = np.random.normal(1, 0.5, size= N_samples)
    theta = np.pi / 2

    params = [
        {
            'N': N,
            'M': M,
            'delta': deltas[i],
            'q': qs[q_ids[i]],
            'mu': mus[i],
            'J': Js[i],
            'delta_q': delta_qs[q_ids[i]],
            't': ts[i],
            'theta': theta
        } for i in range(N_samples)
    ]

    return params

def generate_zm_params(N, M, N_samples, MLpredictor = None):
    params = []
    i = 0
    while(len(params) < N_samples):
        delta = np.random.normal(1.8, 1)
        q = np.random.choice(np.array([np.random.normal(1.8, 0.5), np.random.normal(4.3, 0.5)]))
        mu = np.random.normal(1.8, 1)
        J = np.random.normal(1.8, 1)
        delta_q = np.random.choice(np.array([np.random.normal(0.2, 0.5), np.random.normal(5.9, 0.5)]))
        t = np.random.normal(1, 0.5)
        theta = np.pi / 2
        if MLpredictor is not None:
            X = [{'N':N, 'M':M, 'delta':delta, 'q':q, 'mu':mu, 'J':J, 'delta_q':delta_q, 't':t,  'theta':theta}]
            X = pd.DataFrame(X)
            Y = MLpredictor.predict(X)[0]
        else:
            ladder = SpinLadder(N, M, mu, delta, J, q, delta_q, t, S,  theta=theta)
            num_zm = count_mzm_states(ladder.H, threshold=1.e-5)
            Y = num_zm > 0
        if Y:
            i += 1
            if i % 100 == 0:
                print(f'{i} zero modes generated')
            params.append(
                {
                    'N': N,
                    'M': M,
                    'delta': delta,
                    'q': q,
                    'mu': mu,
                    'J': J,
                    'delta_q': delta_q,
                    't': t,
                    'theta': theta
                }
            )
    return params

if __name__ == '__main__':
    N = 70
    M = 2

    N_samples = 1000000
    N_qs = 1000

    # generate_param_data(N, M, N_samples, './data/spin_ladder/spin_ladder_70_2_red_dist.csv')
    
    # ML_predictor = pickle.load(open(os.path.join(MODEL_SAVE_DIR, MODEL_NAME + '.pkl'), 'rb'))
    # params = generate_zm_params(N, M, N_samples // 2, ML_predictor) + generate_params(N, M, N_samples // 2)
    params = generate_params(N, M, N_samples, N_qs)
    #generate_data(SpinLadder, params, './data/spin_ladder/70_2_RedDistFixedStd', eig_decomposition=True)
    generate_data(SpinLadder, params, './data/spin_ladder/70_2_RedDist1000q', eig_decomposition=False, format='csr')


    # ladder = SpinLadder(**DEFAULT_PARAMS)
    # plot_majorana_polarization(ladder, './plots/spin_ladder/polarization_total', polaxis='total', string_num=2)
    # for c in range(4):
    #     plot_eigvec(ladder.get_hamiltonian(), c, f'./plots/spin_ladder/eigvec_{c}', threshold=1.e-5, string_num=2)
