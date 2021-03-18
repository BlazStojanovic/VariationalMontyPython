"""	
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 14/03/2021
----------
Contains: 
	The harmonium molecule case. k1 = 1/2, k2 = 1.0
"""

"""
Optimizing variance for different jastrow factors

"""
from Utilities.Wavefunction import WaveFunction as wf
from Utilities.Wavefunction import Jastrow as js
from Utilities.Wavefunction import GaussianBasisS as gbf
from Utilities.Wavefunction import LCGOData as orbitals

from Utilities.MonteCarlo import MCMC as mc

import jax
import jax.random as rnd
import jax.numpy as jnp
import jax.lax as lax
from jax import jit, grad, pmap, vmap, jacfwd, jacrev

import numpy as np
from jax.experimental import loops

def VMC(it, jastParam, bparam, c, a):
	bparam = jnp.sqrt(jnp.square(bparam)) # Aid in optimization?
	k1 = 1 # He
	k2 = 1 # H

	Ra = jnp.array([a, 0.0, 0.0])

	tau = 0.3
	seed = 42
	key = rnd.PRNGKey(seed)
	config = np.random.random((2, 3)) # TODO fix later, but for now start at origin
	configurations = jnp.zeros((2*it, 3))
	Els = jnp.zeros(it)

	# define trial wave function
	@jit
	def twf(r1, r2, jastParam, bparam, c):
		d = r1-r2
		r12 = jnp.linalg.norm(d)
		r1h = jnp.linalg.norm(Ra-r1)
		r1He = jnp.linalg.norm(r1)
		r2h = jnp.linalg.norm(Ra-r2)
		r2He = jnp.linalg.norm(r2)

		c1, c2 = c[0], c[1] # coef params of each center
		bp1, bp2 = bparam[0], bparam[1] # exponent params of each center

		b1 = jnp.sum(c1*gbf.evaluate(r1, jnp.zeros((1, 3)), bp1)) + jnp.sum(c2*gbf.evaluate(r1, Ra, bp2))
		b2 = jnp.sum(c1*gbf.evaluate(r2, jnp.zeros((1, 3)), bp1)) + jnp.sum(c2*gbf.evaluate(r2, Ra, bp2))
		
		b = jastParam

		jast = js.HeHstrow(r12, r1h, r1He, r2h, r2He, b)

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
		V = -k1*jnp.reciprocal(jnp.linalg.norm(r1)) - k1*jnp.reciprocal(jnp.linalg.norm(r2)) # He
		V += -k2*jnp.square(jnp.linalg.norm(Ra-r1)) - k2*jnp.reciprocal(jnp.linalg.norm(Ra-r2)) #H
		C = jnp.reciprocal(jnp.linalg.norm(r1-r2))
		Cn = jnp.reciprocal(jnp.linalg.norm(Ra))*k1*k2
		psii = twf(r1, r2, jastParam, bparam, c)
		Hl = T/psii + V + C + Cn

		Els = jax.ops.index_update(Els, i, Hl)

		return key, config, jastParam, bparam, c, Els

	key, config, jastParam, bparam, c, Els = lax.fori_loop(0, it, loop_bdy, (key, config, jastParam, bparam, c, Els))
	
	Ev = jnp.average(Els)
	stdev = jnp.std(Els)*(it)/(it-1)

	return Ev, stdev

VMC = jit(VMC, static_argnums=[0])

def update_sigJ(it, jastParam, bparam, c, a, alpha):
	dE, dsig = jacfwd(VMC, argnums=[1])(it, jastParam, bparam, c, a)
	
	dj = dsig
	return jastParam-alpha*dj

def update_sigB(it, jastParam, bparam, c, a, alpha):
	dE, dsig = jacfwd(VMC, argnums=[2])(it, jastParam, bparam, c, a)
	
	db = dsig
	return bparam-alpha*db

def update_sigC(it, jastParam, bparam, c, a, alpha):
	dE, dsig = jacfwd(VMC, argnums=[3])(it, jastParam, bparam, c, a)
	
	dc = dsig
	return c - alpha*dc


def update_sigA(it, jastParam, bparam, c, a, alpha):
	dE, dsig = jacfwd(VMC, argnums=[4])(it, jastParam, bparam, c, a)
	
	da = dsig
	return a - alpha*da


def update_sig(it, jastParam, bparam, c, a, alpha):
	dE, dsig = jacfwd(VMC, argnums=[1,2,3,4])(it, jastParam, bparam, c, a)
	
	dj, db, dc, da = dsig
	return jastParam-alpha*dj, bparam-alpha*db, c - alpha*dc#, a - alpha*da

update_sigJ = jit(update_sigJ, static_argnums=[0])
update_sigB = jit(update_sigB, static_argnums=[0])
update_sigA = jit(update_sigA, static_argnums=[0])
update_sig = jit(update_sig, static_argnums=[0])

if __name__ == '__main__':
	epochs = 100
	alpha = 0.005

	Evs = np.zeros(epochs)
	sigs = np.zeros(epochs)

	# MC params
	it = 1000
	beta = 1e-3

	N = 10
	# A = np.linspace(0.5, 1.0, N)
	A = [0.7414]
	EoptA = np.zeros(N)
	SigoptA = np.zeros(N)

	bparam = jnp.array([[[0.3425250914E+01], [0.6239137298E+00], [0.1688554040E+00]], 
			[[0.3425250914E+01], [0.6239137298E+00], [0.1688554040E+00]]])
		
	ci = jnp.array([[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
			[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]])

	jastParam = jnp.array([1.0, 1.0 ,1.0, 1.0, 1.0])
	# out = VMC(it, jastParam, bparam, ci, a)

	# print(out)

	import matplotlib.pyplot as plt

	for j,a in enumerate(A):
	# Wave function parameters

		bparam = jnp.array([[[0.6362421394E+01], [0.1158922999E+01],[0.3136497915E+00]], 
			[[0.3425250914E+01], [0.6239137298E+00], [0.1688554040E+00]]])
		
		ci = jnp.array([[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00],
			[0.1543289673E+00, 0.5353281423E+00, 0.4446345422E+00]])

		jastParam = jnp.array([1.0, 1.0 ,1.0, 1.0, 1.0])
		
		for i in range(epochs):
			Evs[i], sigs[i] = VMC(it, jastParam, bparam, ci, a)

			alpex = alpha*jnp.exp(-beta*i)
			# jastParam, bparam, ci, a = update_sig(it, jastParam, bparam, ci, a, alpex)
			jastParam, bparam, ci = update_sig(it, jastParam, bparam, ci, a, alpex)
			
			print("After iteration: ", i)
			print("E: {} +- {}".format(Evs[i], sigs[i]))
			print("Jastrow parameters ", jastParam)
			print("Basis parameters ", bparam)
			print("Expansion Coefs ", ci)
			print("Geometry ", a)
			print()

		plt.plot(Evs)
		plt.fill_between(jnp.arange(len(Evs)), Evs + sigs, Evs - sigs, alpha=0.3)
		plt.show()

		EoptA[j] = Evs[-1]
		EoptA[j] = sigs[-1]

	# np.save("../data/vmcEnergies/vmcM-opt-1g-1j-E.npy", sigs)
	# np.save("../data/vmcEnergies/vmcM-opt-1g-1j-S.npy", Evs)

	plt.plot(EoptA)
	plt.fill_between(jnp.arange(len(EoptA)), EoptA + SigoptA, EoptA - SigoptA, alpha=0.3)
	plt.show()