import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

class KitaevChain():
    def __init__(self, t, mu, delta, N):
        self.t = t
        self.mu = mu
        self.delta = delta
        self.H = self.construct_open_boundary_hamiltonian(N)
        self.H_block = self.construct_open_boundary_hamiltonian_site_blocks(N)
        self.H_csr = self.construct_open_boundary_hamiltonian_csr(N)
        self.H_csr_block = self.construct_open_boundary_hamiltonian_csr_site_blocks(N)

    # Spin block formalism (blocks are formed for different sites for a given spin)
    # Exemplary for N=2
    # mu t delta 0
    # t* -mu 0 -delta*
    # delta* 0 mu t
    # 0 -delta -t* -mu
    def construct_open_boundary_hamiltonian(self, N):
        H = np.zeros((2*N, 2*N), dtype=np.complex128)
        for i in range(2*N):              
            if i > 0:
                if i < N:
                    H[i-1, i] = self.t
                    H[i, i-1] = np.conjugate(self.t)
                if i > N:
                    H[i-1, i] = -self.t
                    H[i, i-1] = -np.conjugate(self.t)
            if i < N:
                H[i, i] = self.mu  
            else:
                H[i, i] = -self.mu
        for i in range(N-1):
            H[i, N+i+1] = self.delta
            H[i+1, N+i] = -np.conjugate(self.delta)

            H[N+i+1, i] = np.conjugate(self.delta)
            H[N+i, i+1] = -self.delta
        return H

    # Site block formalism (blocks are formed for different spins on a given site)
    # Exemplary for N=2
    # mu 0 t delta
    # 0 -mu -delta -t
    # t* -delta* mu 0
    # delta* -t* 0 -mu
    def construct_open_boundary_hamiltonian_site_blocks(self, N):
        block_size = 2
        H = np.zeros((block_size*N, block_size*N), dtype=np.complex128)

        i = 0
        while i < block_size*N - 1:
            # On-site energies
            H[i, i] = self.mu  
            H[i + 1, i + 1] = -self.mu

            # Interactions of neighboring sites
            if i >= block_size:
                H[i - block_size, i] = self.t
                H[i - block_size, i + 1] = self.delta
                H[i - block_size + 1, i] = -self.delta
                H[i - block_size + 1, i + 1] = -self.t

                H[i, i - block_size] = np.conjugate(self.t)
                H[i, i - block_size + 1] = -np.conjugate(self.delta)
                H[i + 1, i - block_size] = np.conjugate(self.delta)
                H[i + 1, i - block_size + 1] = -np.conjugate(self.t)

            i += block_size
        return H
    

    def construct_open_boundary_hamiltonian_csr(self, N):
        ip = 0
        row = np.zeros(10*N-8, dtype=np.int32)
        col = np.zeros(10*N-8, dtype=np.int32)
        data = np.zeros(10*N-8, dtype=np.complex128)
        for i in range(2*N):
            if i > 0:
                if i < N:
                    row[ip] = i-1
                    col[ip] = i
                    data[ip] = self.t
                    ip += 1
                    row[ip] = i
                    col[ip] = i-1
                    data[ip] = np.conjugate(self.t)
                    ip += 1
                if i > N:
                    row[ip] = i-1
                    col[ip] = i
                    data[ip] = -self.t
                    ip += 1
                    row[ip] = i
                    col[ip] = i-1
                    data[ip] = -np.conjugate(self.t)
                    ip += 1
            if i < N:
                row[ip] = i
                col[ip] = i
                data[ip] = self.mu  
                ip += 1
            else:
                row[ip] = i
                col[ip] = i
                data[ip] = -self.mu
                ip += 1

        for i in range(N-1):
            row[ip] = i
            col[ip] = N+i+1
            data[ip] = self.delta
            ip += 1
            row[ip] = i+1
            col[ip] = N+i
            data[ip] = -np.conjugate(self.delta)
            ip += 1

            row[ip] = N+i+1
            col[ip] = i
            data[ip] = np.conjugate(self.delta)
            ip += 1
            row[ip] = N+i
            col[ip] = i+1
            data[ip] = -self.delta
            ip += 1
           
        return csr_matrix((data, (row, col)), shape=(2*N, 2*N), dtype=np.complex128)

    
    def construct_open_boundary_hamiltonian_csr_site_blocks(self, N):
        block_size = 2
        ip = 0
        i = 0
        row = np.zeros(10*N-8, dtype=np.int32)
        col = np.zeros(10*N-8, dtype=np.int32)
        data = np.zeros(10*N-8, dtype=np.complex128)
        while i < (block_size*N) - 1:
            row[ip] = i
            col[ip] = i
            data[ip] = self.mu
            ip += 1
            row[ip] = i + 1
            col[ip] = i + 1
            data[ip] = -self.mu
            ip += 1

            if i >= block_size:
                row[ip] = i - block_size
                col[ip] = i
                data[ip] = self.t
                ip += 1
                row[ip] = i - block_size
                col[ip] = i + 1
                data[ip] = self.delta
                ip += 1
                row[ip] = i - block_size + 1
                col[ip] = i
                data[ip] = -self.delta
                ip += 1
                row[ip] = i - block_size + 1
                col[ip] = i + 1
                data[ip] = -self.t
                ip += 1

                row[ip] = i
                col[ip] = i - block_size
                data[ip] = np.conjugate(self.t)
                ip += 1
                row[ip] = i
                col[ip] = i - block_size + 1
                data[ip] = -np.conjugate(self.delta)
                ip += 1
                row[ip] = i + 1
                col[ip] = i - block_size
                data[ip] = np.conjugate(self.delta)
                ip += 1
                row[ip] = i + 1
                col[ip] = i - block_size + 1
                data[ip] = -np.conjugate(self.t)
                ip += 1

            i += block_size
           
        return csr_matrix((data, (row, col)), shape=(2*N, 2*N), dtype=np.complex128)

t = 1.
mus = np.arange(0., 4*t, 0.1*t)
delta = t
N = 15

for mu in mus:
    kc = KitaevChain(t, mu, delta, N)
    eigvals, eigvecs = np.linalg.eigh(kc.H)
    
    plt.figure()
    plt.scatter(x=np.arange(2*N), y=eigvals)
    plt.xlabel('x')
    plt.ylabel('E')
    plt.savefig('./plots/kitaev/spin_blocks/kitaev_chain_eigvals_mu{:.2}.png'.format(mu))
    plt.close()

    plt.figure()
    plt.title('Eigvals: {:.2}'.format(eigvals[N - 1]))
    plt.plot(np.power(np.abs(eigvecs[:, N-1]), 2))
    plt.xlabel('x')
    plt.ylabel('E')
    plt.savefig('./plots/kitaev/spin_blocks/kitaev_chain_eigvecs_mu{:.2}.png'.format(mu))
    plt.close()

    eigvals, eigvecs = np.linalg.eigh(kc.H_block)
    
    plt.figure()
    plt.scatter(x=np.arange(2*N), y=eigvals)
    plt.xlabel('x')
    plt.ylabel('E')
    plt.savefig('./plots/kitaev/site_blocks/kitaev_chain_eigvals_mu{:.2}.png'.format(mu))
    plt.close()

    plt.figure()
    plt.title('Eigvals: {:.2}'.format(eigvals[N - 1]))
    plt.plot(np.power(np.abs(eigvecs[:, N-1]), 2))
    plt.xlabel('x')
    plt.ylabel('E')
    plt.savefig('./plots/kitaev/site_blocks/kitaev_chain_eigvecs_mu{:.2}.png'.format(mu))
    plt.close()

    k = N
    eigvals_csr, eigvecs_csr = eigsh(kc.H_csr, k=k, which='SM', return_eigenvectors=True)
    eigvals_csr = np.sort(eigvals_csr)
    min_indx = np.argmin(np.abs(eigvals_csr))

    plt.figure()
    plt.scatter(x=np.arange(k), y=eigvals_csr)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.savefig('./plots/kitaev/spin_blocks/kitaev_chain_eigvals_csr_mu{:.2}.png'.format(mu))
    plt.close()

    plt.figure()
    plt.title('Eigvals: {:.2}'.format(eigvals_csr[min_indx]))
    plt.plot(np.power(np.abs(eigvecs_csr[:, min_indx]), 2))
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.savefig('./plots/kitaev/spin_blocks/kitaev_chain_eigvecs_csr_mu{:.2}.png'.format(mu))
    plt.close()

    k = N
    eigvals_csr, eigvecs_csr = eigsh(kc.H_csr_block, k=k, which='SM', return_eigenvectors=True)
    eigvals_csr = np.sort(eigvals_csr)
    min_indx = np.argmin(np.abs(eigvals_csr))

    plt.figure()
    plt.scatter(x=np.arange(k), y=eigvals_csr)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.savefig('./plots/kitaev/site_blocks/kitaev_chain_eigvals_csr_mu{:.2}.png'.format(mu))
    plt.close()

    plt.figure()
    plt.title('Eigvals: {:.2}'.format(eigvals_csr[min_indx]))
    plt.plot(np.power(np.abs(eigvecs_csr[:, min_indx]), 2))
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.savefig('./plots/kitaev/site_blocks/kitaev_chain_eigvecs_csr_mu{:.2}.png'.format(mu))
    plt.close()
