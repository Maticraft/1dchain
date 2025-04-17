from src.data.utils import generate_data
from src.hamiltonian.quantum_dots_chain import AtomicUnits, QuantumDotsHamiltonian, generate_parameters

conductance_config = {
    'gamma': 0.1,
    'cmap0': {
        'i':0,
        'j':0,
        'ef_range': (-2./AtomicUnits.Eh, 2./AtomicUnits.Eh),
        'mu_range': (-2./AtomicUnits.Eh, 2./AtomicUnits.Eh),
        'ef_num': 200,
        'mu_num': 200
    },
    'cmap1': {
        'i':0,
        'j':0,
        'mu_range': (-2./AtomicUnits.Eh, 2./AtomicUnits.Eh),
        'mu_num': 200
    }
}


N = 100
parameters = generate_parameters(N, num_verified_majoranas=N)
generate_data(QuantumDotsHamiltonian, parameters, './data/quantum_dots/3dots1level_test_hopping', eig_decomposition=False, conductance_config=None, format='csr')
