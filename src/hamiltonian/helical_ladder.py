import os
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from run.simpleML_training import MODEL_NAME, MODEL_SAVE_DIR
from src.data_utils import Hamiltonian, generate_data
from src.hamiltonian.utils import count_mzm_states, majorana_polarization, calculate_gap, calculate_mzm_main_bands_gap, plot_eigvals, plot_eigvec, plot_majorana_polarization

DEFAULT_PARAMS = {'N': 70, 'M': 2, 'delta': 0.3, 'mu': 0.9, 'J': 1., 'delta_q': np.pi, 't': 1}
MZM_THRESHOLD = 0.05


class SpinLadder(Hamiltonian):
    def __init__(self, N, M, mu = 0.9, delta = 0.3, J = 1, q = np.pi/2, delta_q = np.pi, t = 1, S = 1, theta = np.pi/2, B = 0., periodic = False, use_disorder = False, increase_potential_at_edges = False, **kwargs):
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
        if periodic:
            self.H = self._add_N_periodic_boundary(self.H)
        if use_disorder:
            V_dis = kwargs.get('disorder_potential', 1.)
            V_pos = kwargs.get('disorder_positions', [{'i': 0, 'j': 0}, {'i': N-1, 'j': M-1}])
            for pos in V_pos:
                self.H = self._add_potential_gate(self.H, pos['i'], pos['j'], V_dis)
        if increase_potential_at_edges:
            V = kwargs.get('potential', 1.)
            before = kwargs.get('potential_before', 0)
            after = kwargs.get('potential_after', self.N)
            self._increase_potential(V, before, after)

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
    

    def _add_N_periodic_boundary(self, H):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        N = self.N
        M = self.M
        for j in range(M):
            H[self._idx(N-1, j) : self._idx(N-1, j) + self.block_size, self._idx(0, j) : self._idx(0, j) + self.block_size] = np.kron(sigma_z, self._interaction_block_ij_cdagger_c(self.t))
            H[self._idx(0, j) : self._idx(0, j) + self.block_size, self._idx(N-1, j) : self._idx(N-1, j) + self.block_size] = np.conjugate(np.kron(sigma_z, self._interaction_block_ij_cdagger_c(self.t))).T
        return H
    

    def _increase_potential(self, V, before=None, after=None):
        if before is None:
            before = 0
        if after is None:
            after = self.N

        positions_before = [{'i': i, 'j': j} for i in range(0, before) for j in range(self.M)]
        positions_after = [{'i': i, 'j': j} for i in range(after, self.N) for j in range(self.M)]
        for pos in positions_before:
            self.H = self._add_potential_gate(self.H, pos['i'], pos['j'], V)
        for pos in positions_after:
            self.H = self._add_potential_gate(self.H, pos['i'], pos['j'], V)
    

    def _add_potential_gate(self, H, i, j, V):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        eye = np.eye(2, dtype=np.complex128)
        block = np.kron(sigma_z, eye)
        H[self._idx(i, j) : self._idx(i, j) + self.block_size, self._idx(i, j) : self._idx(i, j) + self.block_size] += V*block
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
        mp = majorana_polarization(self.H, threshold=MZM_THRESHOLD, axis='y', site='all')
        values = list(mp.values())
        mp_y_sum_left = sum(values[:len(values)//2])
        mp_y_sum_right = sum(values[len(values)//2:])
        mp_tot = majorana_polarization(self.H, threshold=MZM_THRESHOLD, axis='total', site='all')
        values_tot = list(mp_tot.values())
        mp_tot_sum_left = sum(values_tot[:len(values_tot)//2])
        mp_tot_sum_right = sum(values_tot[len(values_tot)//2:])
        band_gap = calculate_gap(self.H)
        num_mzm = count_mzm_states(self.H, threshold=MZM_THRESHOLD)
        if num_mzm > 0:
            mzm_gap = calculate_mzm_main_bands_gap(self.H, mzm_threshold=MZM_THRESHOLD)
        else:
            mzm_gap = 0
        return f"{mp_tot_sum_left}, {mp_tot_sum_right}, {mp_y_sum_left}, {mp_y_sum_right}, {num_mzm}, {band_gap}, {mzm_gap}"



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


def generate_params(N, M, N_samples, periodic=False, use_disorder=False, increase_potential_at_edges=False):
    deltas = np.random.normal(1.8, 1, size= N_samples)
    qs = np.random.uniform(0, 2*np.pi, size = N_samples)
    delta_qs = np.random.uniform(0, 2*np.pi, size = N_samples)
    mus = np.random.normal(1.8, 1, size= N_samples)
    Js = np.random.normal(1.8, 1, size= N_samples)
    ts = np.random.normal(1, 0.5, size= N_samples)
    theta = np.pi / 2

    if use_disorder:
        use_disorder_potential = np.random.choice([True, False], size=N_samples)
        V_dis = np.random.normal(2, 2, size= N_samples)
        num_gates = np.random.randint(1, 10, size=N_samples)
        V_pos_i = [np.random.randint(0, N*M, size= num_gates[i]) for i in range(N_samples)]
    else:
        use_disorder_potential = [False for _ in range(N_samples)]
        V_dis = [0 for _ in range(N_samples)]
        V_pos_i = [[] for _ in range(N_samples)]

    if increase_potential_at_edges:
        use_edge_potential = np.random.choice([True, False], size=N_samples)
        Vs = np.random.normal(5, 5, size= N_samples)
        before_site = np.random.randint(0, N//3, size=N_samples)
        after_site = np.random.randint(2*N//3, N, size=N_samples)
    else:
        use_edge_potential = [False for _ in range(N_samples)]
        Vs = [0 for _ in range(N_samples)]
        before_site = [0 for _ in range(N_samples)]
        after_site = [N for _ in range(N_samples)]


    params = [
        {
            'N': N,
            'M': M,
            'delta': deltas[i],
            'q': qs[i],
            'mu': mus[i],
            'J': Js[i],
            'delta_q': delta_qs[i],
            't': ts[i],
            'theta': theta,
            'periodic': int(periodic),
            'use_disorder': int(use_disorder_potential[i]),
            'disorder_potential': V_dis[i],
            'disorder_positions': [{'i': int(pos // M), 'j': int(pos % M)} for pos in V_pos_i[i]],
            'increase_potential_at_edges': int(use_edge_potential[i]),
            'potential': Vs[i],
            'potential_before': int(before_site[i]),
            'potential_after': int(after_site[i])
        } for i in range(N_samples)
    ]

    return params

def generate_bal_zm_params(N, M, N_samples, MLpredictor = None, periodic=False, use_disorder=False, increase_potential_at_edges=False):
    all_params = []
    num_mzms = 0
    num_not_mzms = 0
    with tqdm(total=N_samples, desc='Generating data params') as pbar:
        while(len(all_params) < N_samples):
            params = generate_random_single_hamiltonian_params(M, N, periodic=periodic, use_disorder=use_disorder, increase_potential_at_edges=increase_potential_at_edges)
            if MLpredictor is not None:
                ml_predictor_keys = ['N', 'M', 'delta', 'q', 'mu', 'J', 'delta_q', 't', 'theta']
                X = [params[key] for key in ml_predictor_keys]
                X = pd.DataFrame(X)
                Y = MLpredictor.predict(X)[0]
            else:
                ladder = SpinLadder(**params)
                num_zm = count_mzm_states(ladder.H, threshold=MZM_THRESHOLD)
                Y = num_zm > 0
            if Y:
                num_mzms += 1
                all_params.append(params)
                pbar.update(1)
            elif num_not_mzms < N_samples // 2:
                num_not_mzms += 1
                all_params.append(params)
                pbar.update(1)
    return all_params


def generate_random_single_hamiltonian_params(M, N, periodic=False, use_disorder=False, increase_potential_at_edges=False):
    delta = np.random.normal(1.8, 1)
    q = np.random.uniform(0, 2*np.pi)
    mu = np.random.normal(1.8, 1)
    J = np.random.normal(1.8, 1)
    delta_q = np.random.uniform(0, 2*np.pi)
    t = np.random.normal(1, 0.5)
    theta = np.pi / 2

    if use_disorder:
        use_disorder_potential = np.random.choice([True, False])
        V_dis = np.random.normal(2, 2)
        num_gates = np.random.randint(1, 10)
        V_pos_i = np.random.randint(0, N*M, size= num_gates)
    else:
        use_disorder_potential = False
        V_dis = 0
        V_pos_i = []

    if increase_potential_at_edges:
        use_edge_potential = np.random.choice([True, False])
        Vs = np.random.normal(5, 5)
        before_site = np.random.randint(0, N//3)
        after_site = np.random.randint(2*N//3, N)
    else:
        use_edge_potential = False
        Vs = 0
        before_site = 0
        after_site = N

    return {
        'N': N,
        'M': M,
        'delta': delta,
        'q': q,
        'mu': mu,
        'J': J,
        'delta_q': delta_q,
        't': t,
        'theta': theta,
        'periodic': int(periodic),
        'use_disorder': int(use_disorder_potential),
        'disorder_potential': V_dis,
        'disorder_positions': [{'i': int(pos // M), 'j': int(pos % M)} for pos in V_pos_i],
        'increase_potential_at_edges': int(use_edge_potential),
        'potential': Vs,
        'potential_before': int(before_site),
        'potential_after': int(after_site)
    }


if __name__ == '__main__':
    N = 70
    M = 2

    N_samples = 1000000
    N_qs = 100

    # generate_param_data(N, M, N_samples, './data/spin_ladder/spin_ladder_70_2_red_dist.csv')
    
    # ML_predictor = pickle.load(open(os.path.join(MODEL_SAVE_DIR, MODEL_NAME + '.pkl'), 'rb'))
    # params = generate_bal_zm_params(N, M, N_samples, periodic=True, use_disorder=False, increase_potential_at_edges=True)
    params = generate_params(N, M, N_samples, periodic=True, use_disorder=False, increase_potential_at_edges=True)
    #generate_data(SpinLadder, params, './data/spin_ladder/70_2_RedDistFixedStd', eig_decomposition=True)
    generate_data(SpinLadder, params, './data/spin_ladder/70_2_RedDistSimplePeriodicPGBalancedZM', eig_decomposition=False, format='csr')


    # ladder = SpinLadder(**DEFAULT_PARAMS)
    # plot_majorana_polarization(ladder, './plots/spin_ladder/polarization_total', polaxis='total', string_num=2)
    # for c in range(4):
    #     plot_eigvec(ladder.get_hamiltonian(), c, f'./plots/spin_ladder/eigvec_{c}', threshold=1.e-5, string_num=2)
