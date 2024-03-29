"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 23/02/2021
----------
Contains: 
	Optimization of energy/variance of Hookium VMC
	
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

# @jit
def VMC(it, jastParam, bparam, c):
	bparam = jnp.sqrt(jnp.square(bparam)) # Aid in optimization?

	tau = 0.3
	seed = 42
	key = rnd.PRNGKey(seed)
	k = 0.25
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

		b1 = jnp.sum(ci*gbf.evaluate(r1, jnp.zeros((1, 3)), bparam))
		b2 = jnp.sum(ci*gbf.evaluate(r2, jnp.zeros((1, 3)), bparam))
		
		a = 1.0/2.0 # opposite spin electron cusp condition

		b = jnp.abs(jastParam)
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

	return Ev, stdev

VMC = jit(VMC, static_argnums=[0])


def update_E(it, jastParam, bparam, c, alpha):
	dEda, dsigda = jacfwd(VMC, argnums=[1, 2])(it, jastParam, bparam, c)
	
	dj, db = dEda
	return jastParam-alpha*dj, bparam-alpha*db


def update_sig(it, jastParam, bparam, c, alpha):
	dE, dsig = jacfwd(VMC, argnums=[1, 2])(it, jastParam, bparam, c)
	
	dj, db = dsig

	return jastParam-10*alpha*dj, bparam-alpha*db

update_sig = jit(update_sig, static_argnums=[0])
update_E = jit(update_E, static_argnums=[0])

if __name__ == '__main__':
	# Optimise variance
	# epochs = 600
	# alpha = 0.5

	# Evs = np.zeros(epochs)
	# sigs = np.zeros(epochs)

	# # MC params
	# it = 10000
	# beta = 10**(-2)

	# # Wave function parameters
	# bparam = jnp.array([[1.0]])
	# jastParam = jnp.array([1.0])
	
	# ci = jnp.array([1.0])

	# for i in range(epochs):
	# 	Ev, stdev = VMC(it, jastParam, bparam, ci)
	# 	print("Variational energy: E_V = {}".format(Ev))
	# 	print("Variance: sigma_e = {}".format(stdev))

	# 	Evs[i], sigs[i] = VMC(it, jastParam, bparam, ci)

	# 	alpex = alpha*jnp.exp(-beta*i)
	# 	jastParam, bparam = update_sig(it, jastParam, bparam, ci, alpex)
	# 	print(i, jastParam, bparam)

	# np.save("../data/vmcEnergies/vmc-opt-1g-1j-ss1.npy", sigs)
	# np.save("../data/vmcEnergies/vmc-opt-1g-1j-sE1.npy", Evs)
	
	# Optimise Variational energy
	epochs = 600
	alpha = 0.05

	Evs = np.zeros(epochs)
	sigs = np.zeros(epochs)

	# MC params
	it = 10000
	beta = 10**(-3)

	# Wave function parameters
	bparam = jnp.array([[0.5]])
	jastParam = jnp.array([0.5])
	
	ci = jnp.array([1.0])

	for i in range(epochs):
		Ev, stdev = VMC(it, jastParam, bparam, ci)
		print("Variational energy: E_V = {}".format(Ev))
		print("Variance: sigma_e = {}".format(stdev))

		Evs[i], sigs[i] = VMC(it, jastParam, bparam, ci)

		alpex = alpha*jnp.exp(-beta*i)
		jastParam, bparam = update_E(it, jastParam, bparam, ci, alpex)
		print(i, jastParam, bparam)

	np.save("../data/vmcEnergies/vmc-opt-1g-1j-Es1.npy", sigs)
	np.save("../data/vmcEnergies/vmc-opt-1g-1j-EE1.npy", Evs)