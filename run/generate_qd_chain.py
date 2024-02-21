import src.hamiltonian.quantum_dots_chain as qd_chain

defaults = qd_chain.DefaultParameters()
parameters = qd_chain.QuantumDotsHamiltonianParameters(no_dots=7, no_levels=2, default_parameters=defaults)
system = qd_chain.QuantumDotsHamiltonian(parameters)
plot = qd_chain.Plotting()

hamiltonian = system.full_hamiltonian()
plot.plot_hamiltonian(hamiltonian)

# eigs = eigh(hamiltonian, eigvals_only=True)

paramsweep, eigenvalues, occupations = system.parameter_sweeping(parameter_name='mu', start=-1., stop=1., num=101)
plot.plot_eigenvalues(paramsweep, eigenvalues, occupations, range=[-1.,1.])