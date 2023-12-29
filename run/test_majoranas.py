import numpy as np
from src.hamiltonian.helical_ladder import DEFAULT_PARAMS, SpinLadder
from src.hamiltonian.utils import plot_eigvals, plot_majorana_polarization

params = DEFAULT_PARAMS.copy()
params['q'] = np.pi
params['increase_potential_at_edges'] = True
params['potential_before'] = 15
params['potential_after'] = 65
params['potential'] = 1
params['periodic'] = True
params['use_potential_gates'] = False
params['potential_positions'] = [{'i': 10, 'j': 0}, {'i': 10, 'j': 1}]

x_axis = 'q'
x_values = np.concatenate((np.arange(0., np.pi, 2*np.pi / 100), np.arange(np.pi, 2*np.pi, 2*np.pi / 100)))


h = SpinLadder(**params)
print(h.get_label())
plot_majorana_polarization(h, './test_majoranas', string_num=2, polaxis='y', threshold=0.05)
plot_eigvals(SpinLadder, x_axis, x_values, params, './test_majoranas/eigvals.png', xnorm=np.pi, ylim=(-2, 2))