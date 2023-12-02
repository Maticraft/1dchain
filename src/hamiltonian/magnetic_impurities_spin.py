import numpy as np
import matplotlib.pyplot as plt

class SpinChain():
    def __init__(self, N, mu, delta, J, q, t = 1, a = 1, S = 1, theta = np.pi/2):
        self.block_size = 4
        self.t = t
        self.mu = mu
        self.delta = delta
        self.a = a
        self.q = q
        self.J = J
        self.S = S
        self.theta = theta
        self.H = self.construct_open_boundary_hamiltonian(N)

    def construct_open_boundary_hamiltonian(self, N):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        
        H = np.zeros((self.block_size*N, self.block_size*N), dtype=np.complex128)
        for i in range(N):
            H[self.idx(i):self.idx(i+1), self.idx(i):self.idx(i+1)] = self.spin_block_i_cdagger_c(self.mu, i, self.theta, self.a, self.q, self.S)

            H[self.idx(i):self.idx(i)+2, self.idx(i)+2:self.idx(i+1)] += self.delta_block_i_cdagger_cdagger(self.delta)
            H[self.idx(i)+2:self.idx(i+1), self.idx(i):self.idx(i)+2] += np.conjugate(self.delta_block_i_cdagger_cdagger(self.delta)).T

            if i < N-1:
                H[self.idx(i):self.idx(i+1), self.idx(i+1):self.idx(i+2)] = np.kron(sigma_z, self.interaction_block_ij_cdagger_c(self.t))
                H[self.idx(i+1):self.idx(i+2), self.idx(i):self.idx(i+1)] = np.conjugate(np.kron(sigma_z, self.interaction_block_ij_cdagger_c(self.t))).T

        return H

    def idx(self, i):
        return self.block_size*i

    def spin_block_i_cdagger_c(self, mu, i, theta, a, q, S):
        sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        mu_block = -mu*np.eye(2, dtype=np.complex128)
        
        block = np.kron(sigma_z, mu_block)
        
        phi = i*a*q
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


if __name__ == '__main__':
    N = 70
    delta = 0.3
    mu = 1.5
    J = 1

    qs = np.linspace(0, np.pi, 100)
    energies = []

    for q in qs:
        chain = SpinChain(N, mu, delta, J, q)
        energies.append(np.linalg.eigvalsh(chain.H))


    plt.ylim(-0.3, 0.3)
    plt.plot(qs/(2*np.pi), energies)
    plt.xlabel('q/2π')
    plt.ylabel('Energy')
    plt.savefig('./plots/spin_impurities/magnetic_impurities_spin.png')
