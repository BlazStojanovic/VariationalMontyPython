"""
Ground energies in the written assignment table
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
		w = jnp.square(jnp.abs(twf(pr1, pr2, jastParam, bparam, c)/twf(r1, r2, jastParam, bparam, c))) # Ratio of wavefunction absolute values 
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

	return Ev, stdev, Els

VMC = jit(VMC, static_argnums=[0])

if __name__ == '__main__':

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

	it = 10000000

	# Setup of initial parameters
	sto2ge = orbitals.sto2g_exponents.get('He-s')
	sto2gc = jnp.array([0.1012511, -1.0581347])
	sto2gOj1 = jnp.array([-0.06367405])

	sto4ge = orbitals.sto4g_exponents.get('He-s')
	sto4gc = jnp.array([0.0021113, -0.02267353, 0.15552285, -1.1129586])
	sto4gOj1 = jnp.array([0.32431903])
	
	ge1 = jnp.array([0.25])
	gj1 = jnp.array([0.2020724])
	gj2 = jnp.array([0.22433668, -0.00510384])

	e, s, Els = VMC(it, jnp.array([1.0]), sto2ge, sto2gc)
	Eb = blocking_transform(Els, 20)
	bav = blocking_average(Els, 20)
	Ev = np.average(Els)
	bvar = blocking_var(Els, Ev, 20)
	be, bs = Eb[-1], bvar[-1]

	print("sto2g + J1: {}+-{}".format(e, s), "\n {}+-{}".format(be, bs) )
	e, s, Els = VMC(it, jnp.array([1.0]), sto4ge, sto4gc)
	Eb = blocking_transform(Els, 20)
	bav = blocking_average(Els, 20)
	Ev = np.average(Els)
	bvar = blocking_var(Els, Ev, 20)
	be, bs = Eb[-1], bvar[-1]
	print("sto4g + J1: {}+-{}".format(e, s), "\n {}+-{}".format(be, bs) )
	e, s, Els = VMC(it, sto2gOj1, sto2ge, sto2gc)
	Eb = blocking_transform(Els, 20)
	bav = blocking_average(Els, 20)
	Ev = np.average(Els)
	bvar = blocking_var(Els, Ev, 20)
	be, bs = Eb[-1], bvar[-1]
	print("sto2g + J1 + opt: {}+-{}".format(e, s), "\n {}+-{}".format(be, bs) )
	e, s, Els = VMC(it, sto4gOj1, sto4ge, sto4gc)
	Eb = blocking_transform(Els, 20)
	bav = blocking_average(Els, 20)
	Ev = np.average(Els)
	bvar = blocking_var(Els, Ev, 20)
	be, bs = Eb[-1], bvar[-1]
	print("sto4g + J1 + opt: {}+-{}".format(e, s), "\n {}+-{}".format(be, bs) )
	e, s, Els = VMC(it, gj1, ge1, jnp.array([1.0]))
	Eb = blocking_transform(Els, 20)
	bav = blocking_average(Els, 20)
	Ev = np.average(Els)
	bvar = blocking_var(Els, Ev, 20)
	be, bs = Eb[-1], bvar[-1]
	print("sto1g + J1 + opt: {}+-{}".format(e, s), "\n {}+-{}".format(be, bs) )
	e, s, Els = VMC(it, gj2, ge1, jnp.array([1.0]))
	Eb = blocking_transform(Els, 20)
	bav = blocking_average(Els, 20)
	Ev = np.average(Els)
	bvar = blocking_var(Els, Ev, 20)
	be, bs = Eb[-1], bvar[-1]
	print("sto1g + J2 + opt: {}+-{}".format(e, s), "\n {}+-{}".format(be, bs) )
	