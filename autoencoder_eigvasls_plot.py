import numpy as np

from helical_ladder import  DEFAULT_PARAMS, SpinLadder
from majorana_utils import plot_autoencoder_eigvals, plot_eigvals
from utils import load_autoencoder


autoencoder_dir = './autoencoder/spin_ladder/70_2/10/symmetric_autoencoder_2_2_k2s2'
eigvals_path_autoencoder = './autoencoder/spin_ladder/70_2/10/symmetric_autoencoder_2_2_k2s2/eigvals_spectre_autoencoder.png'
eigvals_path_ref = './autoencoder/spin_ladder/70_2/10/symmetric_autoencoder_2_2_k2s2/eigvals_spectre_ref.png'
epoch = 20
x_axis = 'q'
x_values = np.arange(0., np.pi, 0.1)
xnorm = np.pi
ylim = (-0.5, 0.5)

encoder, decoder = load_autoencoder(autoencoder_dir, epoch)
plot_eigvals(SpinLadder, x_axis, x_values, DEFAULT_PARAMS, eigvals_path_ref, xnorm=xnorm, ylim=ylim)
plot_autoencoder_eigvals(SpinLadder, encoder, decoder, x_axis, x_values, DEFAULT_PARAMS, eigvals_path_autoencoder, xnorm=xnorm, ylim=ylim)