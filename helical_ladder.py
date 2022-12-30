import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

class SpinLadder():
    def __init__(self, N, M, mu = 0.9, delta = 0.3, J = 1, q = np.pi, delta_q = 0, t = 1, S = 1, theta = np.pi/2):
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
        self.H = self.construct_open_boundary_hamiltonian(self.N, self.M)

    def construct_open_boundary_hamiltonian(self, N, M):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        H = np.zeros((self.block_size*N*M, self.block_size*N*M), dtype=np.complex128)
        for i in range(N):
            for j in range(M):

                H[self.idx(i, j) : self.idx(i, j) + self.block_size, self.idx(i, j) : self.idx(i, j) + self.block_size] = self.J*self.spin_block_i_cdagger_c(self.mu, i, j, self.theta, self.delta_q, self.q, self.S)

                k = self.block_size // 2
                H[self.idx(i, j) : self.idx(i, j) + k, self.idx(i, j) + k : self.idx(i, j) + self.block_size] += self.delta_block_i_cdagger_cdagger(self.delta)
                H[self.idx(i, j) + k : self.idx(i, j) + self.block_size, self.idx(i, j) : self.idx(i, j) + k] += np.conjugate(self.delta_block_i_cdagger_cdagger(self.delta)).T

                if i < N-1:
                    H[self.idx(i, j) : self.idx(i, j) + self.block_size, self.idx(i + 1, j) : self.idx(i + 1, j) + self.block_size] = np.kron(sigma_z, self.interaction_block_ij_cdagger_c(self.t))
                    H[self.idx(i + 1, j) : self.idx(i + 1, j) + self.block_size, self.idx(i, j) : self.idx(i, j) + self.block_size] = np.conjugate(np.kron(sigma_z, self.interaction_block_ij_cdagger_c(self.t))).T

                if j < M-1:  
                    H[self.idx(i, j) : self.idx(i, j) + self.block_size, self.idx(i, j + 1) : self.idx(i, j + 1) + self.block_size] = np.kron(sigma_z, self.interaction_block_ij_cdagger_c(self.t))
                    H[self.idx(i, j + 1) : self.idx(i, j + 1) + self.block_size, self.idx(i, j) : self.idx(i, j) + self.block_size] = np.conjugate(np.kron(sigma_z, self.interaction_block_ij_cdagger_c(self.t))).T

        return H

    def idx(self, i, j):
        return (i*self.M + j)*self.block_size

    def spin_block_i_cdagger_c(self, mu, i, j, theta, delta_q, q, S):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        mu_block = -mu*np.eye(2, dtype=np.complex128)
        
        block = np.kron(sigma_z, mu_block)
        
        phi = i*q + j*delta_q
        spin_block = self.get_spin_matrix(theta, phi, S)

        block[0:2, 0:2] += spin_block
        block[2:4, 2:4] += -spin_block.T

        return block


    def delta_block_i_cdagger_cdagger(self, delta):
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        delta_block = 1j*sigma_y*delta
        return delta_block

    
    def interaction_block_ij_cdagger_c(self, t):
        interaction_block = -t*np.eye(2, dtype=np.complex128)
        return interaction_block


    def get_spin_matrix(self, theta, phi, S):
        pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        pauli_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

        spin_matrix = S*(np.sin(theta)*np.cos(phi)*pauli_x + np.sin(theta)*np.sin(phi)*pauli_y + np.cos(theta)*pauli_z)/2
        return spin_matrix


def count_mzm_states(H):
    eigvals = np.linalg.eigvalsh(H)
    return np.sum(np.abs(eigvals) < 1e-10)


def plot_eigvals(N = 70, M = 2, delta = 0.3, mu = 0.9, J = 1., delta_q = np.pi, qs = np.linspace(0, np.pi, 100), filename = './plots/spin_ladder/helical_spin_ladder.png'):
    energies = []
    for q in qs:
        ladder = SpinLadder(N, M, mu, delta, J, q, delta_q)
        energies.append(np.linalg.eigvalsh(ladder.H))

    plt.ylim(-0.5, 0.5)
    plt.plot(qs/(np.pi), energies)
    plt.xlabel('q/π')
    plt.ylabel('Energy')
    plt.savefig(filename)


def generate_data(N, M, N_samples, flename):
    data = pd.DataFrame(columns=['N', 'M', 'delta', 'q', 'mu', 'J', 'delta_q', 't', 'theta', 'mzm_states'])

    deltas = np.random.choice(np.linspace(0, 3, 100), N_samples)
    qs = np.random.choice(np.linspace(0, 2*np.pi, 100), N_samples)
    mus = np.random.choice(np.linspace(0, 3, 100), N_samples)
    Js = np.random.choice(np.linspace(0, 3, 100), N_samples)
    delta_qs = np.random.choice(np.linspace(0, 2*np.pi, 100), N_samples)
    ts = np.random.choice(np.linspace(0, 2, 100), N_samples)
    thetas = np.random.choice(np.linspace(0, 2*np.pi, 100), N_samples)

    for i in tqdm(range(N_samples), desc='Generating data'): 
        ladder = SpinLadder(N, M, mus[i], deltas[i], Js[i], qs[i], delta_qs[i], ts[i], thetas[i])
        num_mzm_states = count_mzm_states(ladder.H)
        data.loc[i] = [N, M, deltas[i], qs[i], mus[i], Js[i], delta_qs[i], ts[i], thetas[i], num_mzm_states]

    data.to_csv(flename, index=False)


N = 70
M = 2

N_samples = 1000000

generate_data(N, M, N_samples, './data/spin_ladder/spin_ladder_70_2.csv')

