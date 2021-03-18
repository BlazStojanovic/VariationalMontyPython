"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 23/02/2021
----------
Contains: 
	Energy of Hookium atom from simple VMC calculation
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

def VMC(it, jastParam, bparam, c, thprop=0.2, nw=1, tau=1.0, seed=4202):
	key = rnd.PRNGKey(seed)
	Els = jnp.zeros(it)
	Ts = jnp.zeros(it)
	Vs = jnp.zeros(it)
	psi = jnp.zeros(it)
	k = 0.25
	config = np.random.random((2, 3)) # TODO fix later, but for now start at origin
	configurations = jnp.zeros((2*it, 3))
	acc = 0

	@jit 
	def true_twf(r1, r2, jastParam, bparam, c):
		d = r1-r2
		# is_zero = jnp.allclose(d, 0.)
		# d = jnp.where(is_zero, jnp.ones_like(d), d)  # replace d with ones if is_zero
		r12 = jnp.linalg.norm(d)
		return 1.0/(2.0*jnp.sqrt(8.0*jnp.power(jnp.pi, 5.0/2.0) + 5.0*jnp.power(jnp.pi, 3.0)))*jnp.exp(-1.0/4.0*(jnp.square(jnp.linalg.norm(r1)) + jnp.square(jnp.linalg.norm(r2))))*(1.0+0.5*r12)

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

		b = jastParam[0]
		jast = js.Jastrow(r12, a, b)

		return b1*b2*jast
		# return b1*b2*(1+0.5*r12)

	dfdr1 = grad(twf, argnums=0)
	hes1 = jacfwd(dfdr1, argnums=0)
	dfdr2 = grad(twf, argnums=1)
	hes2 = jacfwd(dfdr2, argnums=1)

	@jit
	def loop_bdy(i, state):
		key, config, jastParam, bparam, c, acc, configurations, Els, Ts, Vs, psi = state

		key, subkey = rnd.split(key) 

		r1, r2 = config[0], config[1]

		# drift
		# drift = jnp.vstack((dfdr1(r1, r2, jastParam, bparam, c), dfdr2(r1, r2, jastParam, bparam, c)))/twf(r1, r2, jastParam, bparam, c)
		drift = 0.0
		# generate proposal configuration
		pconfig = mc.greenMove(config, drift, tau, subkey)
		
		# Evaluate metropolis factor
		pr1, pr2 = pconfig[0], pconfig[1]
		w = jnp.square(jnp.abs(twf(pr1, pr2, jastParam, bparam, ci)))/jnp.square(jnp.abs(twf(r1, r2, jastParam, bparam, ci))) # Ratio of wavefunction absolute values 
		t = mc.trialProb(pconfig, config, drift, tau)/mc.trialProb(config, pconfig, drift, tau) # Correct for nonsymmetric trial move probability
		w *= t

		# generate random number to check acceptance
		key, subkey = rnd.split(key)
		r = rnd.uniform(subkey, dtype=jnp.float32) # uniform random between 0 and one

		# check for acceptance
		trfun = lambda cp: cp[1]
		fafun = lambda cp: cp[0]
		config = lax.cond(r < w, trfun, fafun, operand=(config, pconfig))
		
		trfun = lambda t: t+1
		fafun = lambda t: t
		acc = lax.cond(r < w, trfun, fafun, operand=acc)
		
		r1, r2 = config[0], config[1]

		psii = twf(r1, r2, jastParam, bparam, c)
		
		T = -0.5*(jnp.trace(hes1(r1, r2, jastParam, bparam, c))/psii + jnp.trace(hes2(r1, r2, jastParam, bparam, c))/psii)
		V = 0.5*k*jnp.square(jnp.linalg.norm(r1)) + 0.5*k*jnp.square(jnp.linalg.norm(r2))
		C = jnp.reciprocal(jnp.linalg.norm(r1-r2))
		Hl = T + V + C

		# configurations = jax.ops.index_update(configurations, i, config[0])
		# configurations = jax.ops.index_update(configurations, it+i, config[1])

		Els = jax.ops.index_update(Els, i, Hl)
		Ts = jax.ops.index_update(Ts, i, T)
		Vs = jax.ops.index_update(Vs, i, V)
		psi = jax.ops.index_update(psi, i, psii)

		return key, config, jastParam, bparam, c, acc, configurations, Els, Ts, Vs, psi

	key, config, jastParam, bparam, c, acc, configurations, Els, Ts, Vs, psi = lax.fori_loop(0, it, loop_bdy, (key, config, jastParam, bparam, c, acc, configurations, Els, Ts, Vs, psi))
	
	return Els, Ts, Vs, configurations, acc/it, psi

if __name__ == '__main__':
	# Define the system and Construct the basis for the trial wave funciton

	# Solve HF problem to get density matrix and C
	nel = 2
	bparam = orbitals.paper_exponents.get('He-s')
	ci = orbitals.paper_coefficients.get('He-s')
	bparam = jnp.array([[0.25]])
	ci = jnp.array([[1.0]])

	print("Variational Monte Carlo")
	print("-"*50)

	thprop = 0.0

	jastParam = [0.5]
	# it = 1000
	# Els, Ts, Vs, configurations, acc, psi = VMC(it, jastParam, bparam, ci, thprop=0.0, nw=1, tau=0.2, seed=4202)

	# jnp.save("../data/vmcConfigurations/vmc-hf-8g-j1_1e3.npy", configurations)

	# print("Acceptance probability: {}".format(acc))

	# print("Variational energy: E_V = {}".format(jnp.average(Els)))

	# print("Variance: sigma_e = {}".format(jnp.std(Els)))

	# it = 10000
	# Els, Ts, Vs, configurations, acc, psi = VMC(it, jastParam, bparam, ci, thprop=0.0, nw=1, tau=0.2, seed=4202)

	# # vmc_g = jacfwd(VMC, argnums=1)
	# # print("Gradient of the function: {}".format(vmc_g(it, jastParam, bparam, ci, thprop=thprop, tau=0.7)))

	# jnp.save("../data/vmcConfigurations/vmc-hf-8g-j1_1e4.npy", configurations)

	# print("Acceptance probability: {}".format(acc))

	# print("Variational energy: E_V = {}".format(jnp.average(Els)))

	# print("Variance: sigma_e = {}".format(jnp.std(Els)))

	it = 1000
	Els, Ts, Vs, configurations, acc, psi = VMC(it, jastParam, bparam, ci, thprop=0.0, nw=1, tau=0.2, seed=4202)

	# vmc_g = jacfwd(VMC, argnums=1)
	# print("Gradient of the function: {}".format(vmc_g(it, jastParam, bparam, ci, thprop=thprop, tau=0.7)))

	# jnp.save("../data/vmcConfigurations/vmc-hf-8g-j1_1e5.npy", configurations)

	print("Acceptance probability: {}".format(acc))

	print("Variational energy: E_V = {}".format(jnp.average(Els)))

	print("Variance: sigma_e = {}".format(jnp.std(Els)))
	
	import matplotlib.pyplot as plt
	
	plt.plot(Els, 'g-', alpha=0.6)

	jnp.save("../data/vmcEnergies/vmc-hf-1g-j1-E2.npy", Els)


	# plt.plot(blocking_transform(Els, 20), 'k-')
	# plt.plot(psi, '-r')
	# plt.plot(Ts, 'b-')
	# plt.plot(Vs, 'm-')
	plt.show()