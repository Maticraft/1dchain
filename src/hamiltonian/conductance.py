import typing as t

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv
import torch

from src.hamiltonian.quantum_dots_chain import AtomicUnits as au
from src.hamiltonian.quantum_dots_chain import QuantumDotsHamiltonian

class Transport:
    def __init__(self, hamiltonian: QuantumDotsHamiltonian, gamma=.1):
        self.h = hamiltonian
        # build W matrix
        self.W = np.zeros((self.h.dim,self.h.dim))
        sg = np.sqrt(gamma/au.Eh)
        site = np.diag([sg,sg,-sg,-sg])
        if self.h.parameters.no_levels > 1:
            site = np.kron(np.eye(self.h.parameters.no_levels), site)
        i = 0  # left lead: (why only the first dot?)
        self.W[i*self.h.dim0:(i+1)*self.h.dim0, i*self.h.dim0:(i+1)*self.h.dim0] += site 
        i = self.h.parameters.no_dots-1  # right lead
        self.W[i*self.h.dim0:(i+1)*self.h.dim0, i*self.h.dim0:(i+1)*self.h.dim0] += site 

    def s_matrix(self, ef, hamiltonian):
        Smat = inv(np.eye(self.h.dim)*ef - hamiltonian + self.W @ np.conjugate(self.W.T)*0.5j)
        Smat = np.conjugate(self.W.T) @ Smat @ self.W
        Smat = np.eye(self.h.dim) - Smat*1.j
        return Smat
    
    def c_ij_(self, i, j, S):
        """
        C = dI_i/dV_j 
        i = {0,1} = {L,R} <- current via L/R lead
        j = {0,1} = {L,R} <- L/R lead (electrode) voltage
        """
        dij = np.eye(2)[i,j]
        i *= self.h.parameters.no_dots-1
        j *= self.h.parameters.no_dots-1
        #
        Sij = S.reshape(np.rint(self.h.dim/4).astype(int),2,2,np.rint(self.h.dim/4).astype(int),2,2)
        r_ee = Sij[i,0,:,j,0,:]
        r_he = Sij[i,1,:,j,0,:]
        return (2.*dij*self.h.parameters.no_levels - np.trace(r_ee @ np.conjugate(r_ee.T)) + np.trace(r_he @ np.conjugate(r_he.T))).real
    
    def c_ij(self, i, j, ef):
        """
        ef = Fermi energy
        """
        s_matrix = self.s_matrix(ef, self.h.get_hamiltonian())
        return self.c_ij_(i, j, s_matrix)
    
    def kitaevH(self, m):
        H = [[m/2, 0.5, 0, 0, -0.5, 0], 
             [0.5, m/2, 0.5, 0.5, 0, -0.5], 
             [0, 0.5, m/2, 0, 0.5, 0], 
             [0, 0.5, 0, -(m/2), -0.5, 0], 
             [-0.5, 0, 0.5, -0.5, -(m/2), -0.5], 
             [0, -0.5, 0, 0, -0.5, -(m/2)]]
        return np.kron(np.array(H).reshape(2,3,2,3).transpose(1,0,3,2).reshape(6,6), np.eye(2)).astype(complex)
    
    def c_ij_test(self, i, j, ef, mu):
        return self.c_ij_(i, j, self.s_matrix(ef, self.kitaevH(mu)))
    
    def c_map0(
        self,
        i:float,
        j:float,
        ef_range: t.Tuple[float, float] = (-2./au.Eh, 2./au.Eh),
        mu_range:t.Tuple[float, float] = (-2./au.Eh, 2./au.Eh),
        ef_num:int = 200,
        mu_num:int = 200
    ):
        c_map = np.zeros((ef_num, mu_num))
        efs = np.linspace(ef_range[0], ef_range[1], num=ef_num, endpoint=True)
        mus = np.linspace(mu_range[0], mu_range[1], num=mu_num, endpoint=True)
        for y, mu in enumerate(mus):
            self.h.set_parameter('mu', mu)
            for x, ef in enumerate(efs):
                c_map[x, y] = self.c_ij(i, j, ef)
        return c_map

    def c_map1(
        self,
        i:float,
        j:float,
        mu_range: t.Tuple[float, float] = (-2./au.Eh, 2./au.Eh),
        mu_num:int = 200
    ):
        assert self.h.parameters.no_dots == 3, "This method is only implemented for 3 dots"
        c_map = np.zeros((mu_num, mu_num))
        mus = np.linspace(mu_range[0], mu_range[1], num=mu_num, endpoint=True)
        for y, mul in enumerate(mus):
            for x, mur in enumerate(mus):
                mu = np.array([mul, 0., mur])
                self.h.set_parameter('mu', mu)
                c_map[x, y] = self.c_ij(i, j, 0.)
        return c_map

    def c_map2(
        self,
        i: float,
        j: float,
        b_range: t.Tuple[float, float] = (0./au.Eh, 2./au.Eh),
        ef_range: t.Tuple[float, float] = (-1./au.Eh, 1./au.Eh),
        b_num: int = 200,
        ef_num: int = 200,
    ):
        c_map = np.zeros((ef_num, b_num))
        bs = np.linspace(b_range[0], b_range[1], num=b_num, endpoint=True)
        efs = np.linspace(ef_range[0], ef_range[1], num=ef_num, endpoint=True)
        for y, b in enumerate(bs):
            self.h.set_parameter('b', b)
            for x, ef in enumerate(efs):
                c_map[x, y] = self.c_ij(i, j, ef)
        return c_map
    

def torch_conductance_map0(
    h_tensor: torch.Tensor, # must be complex and in standard representation
    i: float,
    j: float,
    ef_range: t.Tuple[float, float] = (-2./au.Eh, 2./au.Eh),
    mu_range: t.Tuple[float, float] = (-2./au.Eh, 2./au.Eh),
    ef_num: int = 200,
    mu_num: int = 200,
    n_levels: int = 1,
    gamma: float = 0.1,
) -> torch.Tensor:

    efs = torch.linspace(ef_range[0], ef_range[1], steps=ef_num).to(h_tensor.device)
    mus = torch.linspace(mu_range[0], mu_range[1], steps=mu_num).view(mu_num, *((1,)*len(h_tensor.shape[:-2])), 1).expand(mu_num, *h_tensor.shape[:-2], 1).to(h_tensor.device)
    hs = h_tensor.view(1, *h_tensor.shape).expand(mu_num, *h_tensor.shape)
    hs_modified = torch_h_set_mu(hs, mus, use_dot_split=n_levels > 1)
    return _torch_cmap(hs_modified, efs, i, j, n_levels, gamma)


def torch_conductance_map2(
    h_tensor: torch.Tensor, # must be complex and in standard representation
    i: float,
    j: float,
    ef_range: t.Tuple[float, float] = (-2./au.Eh, 2./au.Eh),
    b_range: t.Tuple[float, float] = (0./au.Eh, 2./au.Eh),
    ef_num: int = 200,
    b_num: int = 200,
    n_levels: int = 1,
    gamma: float = 0.1,
) -> torch.Tensor:

    efs = torch.linspace(ef_range[0], ef_range[1], steps=ef_num).to(h_tensor.device)
    bs = torch.linspace(b_range[0], b_range[1], steps=b_num).view(b_num, *((1,)*len(h_tensor.shape[:-2])), 1).expand(b_num, *h_tensor.shape[:-2], 1).to(h_tensor.device)
    hs = h_tensor.view(1, *h_tensor.shape).expand(b_num, *h_tensor.shape)
    hs_modified = torch_h_set_b(hs, bs, use_dot_split=n_levels > 1)
    return _torch_cmap(hs_modified, efs, i, j, n_levels, gamma)


def _torch_cmap(
    h_tensor: torch.Tensor,
    efs: torch.Tensor, 
    i: float,
    j: float,
    n_levels: int = 1,
    gamma: float = 0.1
) -> torch.Tensor:

    w_matrix = torch_w_matrix(h_tensor.shape[-1], gamma, n_levels)
    s_matrix = torch_s_matrix(h_tensor, efs.view(efs.shape[0], *((1,)*len(h_tensor.shape))), w_matrix)

    dij = torch.eye(2)[i,j]
    n_dots = h_tensor.shape[-1] // (4 * n_levels)
    i *= n_dots - 1
    j *= n_dots - 1

    Sij = s_matrix.reshape(*s_matrix.shape[:-2], s_matrix.shape[-1] // 4, 2, 2, s_matrix.shape[-1] // 4, 2, 2)
    r_ee = Sij[..., i, 0, :, j, 0, :]
    r_he = Sij[..., i, 1, :, j, 0, :]
    
    trace_vmapped = nest_vmap(torch.trace, len(r_he.shape) - 2)
    c_map = (2.*dij*n_levels - trace_vmapped(r_ee @ torch.conj(r_ee.transpose(-2, -1))) + trace_vmapped(r_he @ torch.conj(r_he.transpose(-2, -1)))).real
    return c_map.moveaxis(0, -1).moveaxis(0, -1)
    

def nest_vmap(func, n_dims):
    for _ in range(n_dims):
        func = torch.vmap(func)
    return func


def torch_h_set_mu(
    h_tensor: torch.Tensor,
    mu: torch.Tensor, # should be either a tensor of shape (..., 1) or (..., n_blocks)
    use_dot_split: bool = False
) -> torch.Tensor:
    assert h_tensor.shape[-1] >= 4, "At least one 4 x 4 block is required"
    n_blocks = h_tensor.shape[-1] // 4
    if mu.shape[-1] == 1:
        mu = mu.expand(*mu.shape[:-1], n_blocks)

    magnetic_field = torch.stack(
        [(h_tensor[..., i*4, i*4] + h_tensor[..., i*4 + 3, i*4 + 3]) / 2 for i in range(n_blocks)],
        dim=-1
    )

    if use_dot_split:
        dot_splits = torch.stack(
            [h_tensor[..., i*4, i*4] - h_tensor[..., (i+1)*4, (i+1)*4] for i in range(n_blocks - 1)],
            dim=-1
        )
        dot_splits = torch.cat([torch.zeros_like(dot_splits[..., :1]), dot_splits], dim=-1)
        mu = mu + dot_splits
    
    diag_idx = torch.arange(h_tensor.shape[-1], device=h_tensor.device)
    modified_h_tensor = h_tensor.clone()
    modified_h_tensor[..., diag_idx[0::4], diag_idx[0::4]] = -mu + magnetic_field
    modified_h_tensor[..., diag_idx[1::4], diag_idx[1::4]] = -mu - magnetic_field
    modified_h_tensor[..., diag_idx[2::4], diag_idx[2::4]] = mu - magnetic_field
    modified_h_tensor[..., diag_idx[3::4], diag_idx[3::4]] = mu + magnetic_field
    return modified_h_tensor


def torch_h_set_b(
    h_tensor: torch.Tensor,
    b: torch.Tensor, # should be either a tensor of shape (..., 1) or (..., n_blocks)
    use_dot_split: bool = False
) -> torch.Tensor:
    assert h_tensor.shape[-1] >= 4, "At least one 4 x 4 block is required"
    n_blocks = h_tensor.shape[-1] // 4
    if b.shape[-1] == 1:
        b = b.expand(*b.shape[:-1], n_blocks)

    mu = torch.stack(
        [(h_tensor[..., i*4 + 2, i*4 + 2] + h_tensor[..., i*4 + 3, i*4 + 3]) / 2 for i in range(n_blocks)],
        dim=-1
    )

    if use_dot_split:
        dot_splits = torch.stack(
            [h_tensor[..., i*4, i*4] - h_tensor[..., (i+1)*4, (i+1)*4] for i in range(n_blocks - 1)],
            dim=-1
        )
        dot_splits = torch.cat([torch.zeros_like(dot_splits[..., :1]), dot_splits], dim=-1)
        mu = mu + dot_splits
    
    diag_idx = torch.arange(h_tensor.shape[-1], device=h_tensor.device)
    modified_h_tensor = h_tensor.clone()
    modified_h_tensor[..., diag_idx[0::4], diag_idx[0::4]] = -mu + b
    modified_h_tensor[..., diag_idx[1::4], diag_idx[1::4]] = -mu - b
    modified_h_tensor[..., diag_idx[2::4], diag_idx[2::4]] = mu - b
    modified_h_tensor[..., diag_idx[3::4], diag_idx[3::4]] = mu + b
    return modified_h_tensor


def torch_w_matrix(
    matrix_dim: int,
    gamma: float = 0.1,
    n_levels: int = 1
) -> torch.Tensor:
    sg = np.sqrt(gamma/au.Eh)
    sg_tensor = torch.tensor([sg, sg, -sg, -sg])
    site = torch.diag(sg_tensor)
    if n_levels > 1:
        site = torch.kron(torch.eye(n_levels), site)

    W = torch.zeros((matrix_dim, matrix_dim))
    dim_0 = 4 * n_levels  # block_size * n_levels
    # left lead
    W[:dim_0, :dim_0] += site 
    # right lead
    W[-dim_0:, -dim_0:] += site 
    return W


def torch_s_matrix(
    h_tensor: torch.Tensor,
    ef: torch.Tensor, # must be of shape matching h_tensor
    w_matrix: torch.Tensor, # of shape (matrix_dim, matrix_dim)
) -> torch.Tensor:
    
    w_matrix = w_matrix.to(h_tensor.device).to(h_tensor.dtype)
    S = torch.eye(h_tensor.shape[-1], device=h_tensor.device, dtype=h_tensor.dtype) * ef - h_tensor + w_matrix @ w_matrix.T * 0.5j
    Smat = torch.linalg.inv(S)
    Smat = w_matrix.T @ Smat @ w_matrix
    Smat = torch.eye(h_tensor.shape[-1], device=h_tensor.device, dtype=h_tensor.dtype) - Smat*1.j
    return Smat


def c_ij_(self, i, j, S):
    """
    C = dI_i/dV_j 
    i = {0,1} = {L,R} <- current via L/R lead
    j = {0,1} = {L,R} <- L/R lead (electrode) voltage
    """
    dij = np.eye(2)[i,j]
    i *= self.h.parameters.no_dots-1
    j *= self.h.parameters.no_dots-1
    #
    Sij = S.reshape(np.rint(self.h.dim/4).astype(int),2,2,np.rint(self.h.dim/4).astype(int),2,2)
    r_ee = Sij[i,0,:,j,0,:]
    r_he = Sij[i,1,:,j,0,:]
    return (2.*dij*self.h.parameters.no_levels - np.trace(r_ee @ np.conjugate(r_ee.T)) + np.trace(r_he @ np.conjugate(r_he.T))).real
    

    

def plot_conductance_map(
    conductance_map: np.ndarray,
    filename: str,
    xtick_range: t.Optional[t.Tuple[float, float]] = None,
    ytick_range: t.Optional[t.Tuple[float, float]] = None,
    xlabel="",
    ylabel=""
):
    fig, ax = plt.subplots()
    im = ax.imshow(conductance_map, origin='lower', extent=[-1, 1, -1, 1])
    if xtick_range is not None:
        x_exponenent = np.floor(np.log10(xtick_range[1]))
        ax.set_xticks(np.linspace(-1, 1, num=5))
        xticklabels = np.linspace(xtick_range[0]*10**(-x_exponenent), xtick_range[1]*10**(-x_exponenent), num=5)
        ax.set_xticklabels(np.round(xticklabels, 2))
        xlabel = xlabel + f' ($10^{{{int(x_exponenent)}}}$)'
    if ytick_range is not None:
        y_exponenent = np.floor(np.log10(ytick_range[1]))
        ax.set_yticks(np.linspace(-1, 1, num=5))
        yticklabels = np.linspace(ytick_range[0]*10**(-y_exponenent), ytick_range[1]*10**(-y_exponenent), num=5)
        ax.set_yticklabels(np.round(yticklabels, 2))
        ylabel = ylabel + f' ($10^{{{int(y_exponenent)}}}$)'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'conductance')
    plt.savefig(filename)
    plt.close()


def generate_conductance_tensor(h_torch: torch.Tensor, cmap_config: t.Dict[str, t.Any]) -> torch.Tensor:
    """
    h_torch: torch.Tensor - must be denormalized complex hamitlonian tensor in standard representation
    """
    if "cmap_list" in cmap_config:
        predicted_cmap = torch.cat(
            [generate_conductance_tensor(h_torch, cmap_subconfig) for cmap_subconfig in cmap_config["cmap_list"]],
            dim=-3
        )
        return predicted_cmap
    if "cmap0" in cmap_config:
        predicted_cmap = torch_conductance_map0(h_torch.unsqueeze(-3), **cmap_config["cmap0"])
    elif "cmap2" in cmap_config:
        predicted_cmap = torch_conductance_map2(h_torch.unsqueeze(-3), **cmap_config["cmap2"])
    return predicted_cmap
