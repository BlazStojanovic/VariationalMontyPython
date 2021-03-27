"""
Correlation energy of different Hookium atoms with different 
"""

from Utilities.Wavefunction import WaveFunction as wf
from Utilities.Wavefunction import Jastrow as js
from Utilities.Wavefunction import GaussianBasisS as gbf
from Utilities.Wavefunction import LCGOData as orbitals

from Utilities.HartreeFock import HartreeFockG as hf

from Utilities.MonteCarlo import MCMC as mc

import jax
import jax.random as rnd
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, grad, pmap, vmap, jacfwd, jacrev

import numpy as np
from jax.experimental import loops

def VMC(it, jastParam, bparam, c, k):
	bparam = jnp.sqrt(jnp.square(bparam)) # Aid in optimization?

	tau = 0.3
	seed = 42
	key = rnd.PRNGKey(seed)
	config = np.random.random((2, 3)) # TODO fix later, but for now start at origin
	configurations = jnp.zeros((2*it, 3))

	Els = jnp.zeros(it)

	# define trial wave function
	@jit
	def twf(r1, r2, jastParam, bparam, c):
		# r1, r2 = config
		d = r1-r2
		# is_zero = jnp.allclose(d, 0.)
		# d = jnp.where(is_zero, jnp.ones_like(d), d)  # replace d with ones if is_zero
		r12 = jnp.linalg.norm(d)
		# r12 = jnp.where(is_zero, 0., r12)  # replace norm with zero if is_zero

		b1 = jnp.sum(c*gbf.evaluate(r1, jnp.zeros((1, 3)), bparam))
		b2 = jnp.sum(c*gbf.evaluate(r2, jnp.zeros((1, 3)), bparam))
		
		a = 1.0/2.0 # opposite spin electron cusp condition
		b = jastParam

		jast = js.Jastrow(r12, a, b)

		return b1*b2*jast

	dfdr1 = jacrev(twf, argnums=0)
	dfdr2 = jacrev(twf, argnums=1)

	hes1 = jacfwd(dfdr1, argnums=0)
	hes2 = jacfwd(dfdr2, argnums=1)

	@jit
	def loop_bdy(i, state):
		key, config, jastParam, bparam, c, Els = state

		key, subkey = rnd.split(key) 

		r1, r2 = config[0], config[1]

		# drift
		drift = jnp.vstack((dfdr1(r1, r2, jastParam, bparam, c), dfdr2(r1, r2, jastParam, bparam, c)))
		
		# generate proposal configuration
		pconfig = mc.greenMove(config, drift, tau, subkey)
		
		# Evaluate metropolis factor
		pr1, pr2 = pconfig[0], pconfig[1]
		w = jnp.square(jnp.abs(twf(pr1, pr2, jastParam, bparam, ci)/twf(r1, r2, jastParam, bparam, ci))) # Ratio of wavefunction absolute values 
		t = mc.trialProb(pconfig, config, drift, tau)/mc.trialProb(config, pconfig, drift, tau) # Correct for nonsymmetric trial move probability
		w *= t

		# generate random number to check acceptance
		key, subkey = rnd.split(key)
		r = rnd.uniform(subkey, dtype=jnp.float32) # uniform random between 0 and one

		# check for acceptance
		trfun = lambda cp: pconfig
		fafun = lambda cp: config
		config = lax.cond(r < w, trfun, fafun, operand=(config, pconfig))
	
		r1, r2 = config
		T = -0.5*(jnp.trace(hes1(r1, r2, jastParam, bparam, c)) + jnp.trace(hes2(r1, r2, jastParam, bparam, c)))
		V = 0.5*k*jnp.square(jnp.linalg.norm(r1)) + 0.5*k*jnp.square(jnp.linalg.norm(r2))
		C = jnp.reciprocal(jnp.linalg.norm(r1-r2))
		psii = twf(r1, r2, jastParam, bparam, c)
		Hl = T/psii + V + C

		Els = jax.ops.index_update(Els, i, Hl)

		return key, config, jastParam, bparam, c, Els

	key, config, jastParam, bparam, c, Els = lax.fori_loop(0, it, loop_bdy, (key, config, jastParam, bparam, c, Els))
	
	Ev = jnp.average(Els)
	stdev = jnp.sqrt(1/it/(it-1)*jnp.sum(jnp.square(Els-Ev)))

	return Ev, stdev#, Els

VMC = jit(VMC, static_argnums=[0])

def update_sig(it, jastParam, bparam, c, alpha, k):
	# dE, dsig, dEls = jacfwd(VMC, argnums=[1, 2])(it, jastParam, bparam, c, k)
	dE, dsig = jacfwd(VMC, argnums=[1, 2])(it, jastParam, bparam, c, k)
	
	dj, db = dsig

	return jastParam-3*alpha*dj, bparam-alpha*db

update_sig = jit(update_sig, static_argnums=[0])

def update(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter, alpha):
	db, (dD, dC) = jacfwd(hf.SCFLoop, argnums=0)(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-6)
	return bparam - alpha*db

def blocking_transform(E, Nb):
	Eb = np.zeros(np.shape(E)[0])
	n = (np.shape(E)[0])//Nb
	# print(n)
	for i in range(n-1):
		Eb[i*Nb:(i+1)*Nb] = np.average(E[i*Nb:(i+1)*Nb])
	Eb[Nb*(n-1)::] = np.average(E[Nb*(n-1)::])
	return Eb

def blocking_average(E, Nb):
	ar = E[::Nb]
	n = (np.shape(E)[0])//Nb
	av = np.cumsum(ar)/(np.arange(n)+1)
	return np.repeat(av, Nb)

def blocking_var(E, Ev, Nb):
	ar = E[::Nb]
	n = (np.shape(E)[0])//Nb
	av = np.cumsum(np.square(ar-Ev))/(np.arange(n))
	return np.repeat(av, Nb)

if __name__ == '__main__':
	
	# Correlation energies for k in range (0.1, 2)
	N = 21

	ks = np.linspace(0.1, 2, N, endpoint=True)

	Evs = np.zeros(N)
	sigvs = np.zeros(N)
	Ehfs = np.zeros(N)

	nel = 2
	centers = ['hooke']
	cpos = jnp.array([[0.0, 0.0, 0.0]])

	ncs = 1

	# Optimization parameters
	hfalpha = 0.05
	hfepochs = 15
	maxiter = 15
	mintol  = 1e-5

	# VMC optimization
	epochs = 300
	alpha = 0.05

	# MC params
	it = 10000
	beta = 5/epochs

	for i, k in enumerate(ks):
		print(i)

		print("Solving for k = ", k)
		bparam = orbitals.sto6g_exponents.get('He-s')
		M = jnp.shape(bparam)[0]
		ccoefs = jnp.array([k])

		# HF optimization
		for epoch in range(hfepochs):
			bparam = update(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter, hfalpha)
			print(bparam)

		Ehfs[i], (D, C) = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-5)
		print("Hartree-Fock energy: ", Ehfs[i])

		ci = C[:, 0]
		if i==0:
			jastParam = jnp.array([0.2])
			vbparam = jnp.array([0.15])
			ci = jnp.array([1.0])

		if i==1:
			epochs = 100

		for j in range(epochs):
			alpex = alpha*jnp.exp(-beta*i)
			jastParam, vbparam = update_sig(it, jastParam, vbparam, ci, alpex, k)
			print(jastParam, vbparam)

		Evs[i], sigvs[i] = VMC(it, jastParam, vbparam, ci, k)

		# Evs[i], s, Els = VMC(it, jastParam, vbparam, ci, k)
		# Eb = blocking_transform(Els, 20)
		# bav = blocking_average(Els, 20)
		# Ev = np.average(Els)
		# bvar = blocking_var(Els, Ev, 20)
		# be, bs = Eb[-1], bvar[-1]
		# sigvs[i] = bs

		print("VMC energy: ", Evs[i], " +-", sigvs[i])

	# np.save("../data/vmcEnergies/corr1-Ehf.npy", Ehfs)
	np.save("../data/vmcEnergies/corr1-Ev.npy", Evs)
	np.save("../data/vmcEnergies/corr1-sigma.npy", sigvs)