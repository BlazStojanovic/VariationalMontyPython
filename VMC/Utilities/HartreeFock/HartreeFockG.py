
"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 24/02/2021
----------
Contains: 
	Fully variational HF with Gaussian basis functions, various implementations of HF
	are split into scripts so there is no hassle with objects and jax (lack of time). 
"""

import jax
import jax.numpy as jnp
from jax import random, jit, vmap, pmap
from Utilities.Wavefunction import GaussianBasisS as gbf
from Utilities.Linalg import eigh
import numpy as np

from jax.config import config
config.update("jax_enable_x64", True)

def initC(M, nel):
	"""
	Parameters
	----------
	M: Number of basis functions used in the expansion
	nel: number of electrons (restricted HF, therefore must be even)

	Returns
	----------
	Initialization of the coefficient matrix

	"""
	key = random.PRNGKey(0)
	C = random.normal(key, (nel, M))
	return C

def constructD(C, M, nel):
	"""
	Construct density matrix from solution to generalized eig problem
	"""
	return 2*jnp.einsum('pa,qa', C[:, :nel//2], C[:, :nel//2])

def constructD0(C, M, nel):
	"""
	Construct density matrix from expansion coefficients
	"""
	return 2*jnp.einsum('ap,aq', C[:nel//2, :], C[:nel//2, :])

def getElDensity(r, D, bpos, bparam, M):
	rho = jnp.zeros(jnp.shape(r)[:-1]) # Assuming jnp.shape(r) = (Npoints, 3)
	for p in range(M):
		for q in range(M):
			rho += D[p, q]*gbf.evaluate(r, bpos[p], bparam[p])*gbf.evaluate(r, bpos[q], bparam[q])
	
	return rho

def constructS(bpos, bparam, M):
	"""
	Construct the overlap matrix
	"""
	S = jnp.zeros((M, M), dtype=jnp.float64)

	for p in range(M):
		for q in range(p, M):
			N = 1.0
			spq = gbf.Spq(bpos[p], bpos[q], bparam[p], bparam[q])
			S = jax.ops.index_update(S, (p, q), N*spq)
			S = jax.ops.index_update(S, (q, p), N*spq)

	return S

@jit
def constructG(D, V2B, bpos, bparam, M):
	G = jnp.einsum('rs,pqrs', D, V2B) - 0.5*jnp.einsum('rs,prsq', D, V2B)

	return G

def constructV2B(bpos, bparam, M):
	"""
	Construct the two-electon tensor (pqrs)

	because the orbitals are real 
	(ij|kl) = (kl|ij) = (ji|lk) = (lk|ji) = (ji|kl) = (lk|ij) = (ij|lk) = (kl|ji), should give ~8x speedup

	"""
	V2B = jnp.zeros((M,M,M,M), dtype=jnp.float64)
	for p in range(M):
		for q in range(p, M):
			for r in range(M):
				for s in range(r, M):
					# print(p, q, r, s)
					v2b = gbf.pqrs(bpos[p], bpos[q], bpos[r], bpos[s], bparam[p], bparam[q], bparam[r], bparam[s])
					V2B = jax.ops.index_update(V2B, (p, q, r, s), v2b)
					V2B = jax.ops.index_update(V2B, (p, q, s, r), v2b)
					V2B = jax.ops.index_update(V2B, (q, p, r, s), v2b)
					V2B = jax.ops.index_update(V2B, (q, p, s, r), v2b)

	return V2B

def constructT(bpos, bparam, M):
	"""
	Construct the kinetic energy matrix
	"""
	T = jnp.zeros((M, M), dtype=jnp.float64)

	for p in range(M):
		for q in range(p, M):
			# Kinetic term
			tpq = gbf.Tpq(bpos[p], bpos[q], bparam[p], bparam[q])
			T = jax.ops.index_update(T, (p, q), tpq)
			T = jax.ops.index_update(T, (q, p), tpq)

	return T

def constructV(bpos, bparam, M, k):
	"""
	Construct the potential energy matrix
	"""
	V = jnp.zeros((M, M), dtype=jnp.float64)
	for p in range(M):
		for q in range(p, M):
			# Potential term
			# Here there should be a loop over all atom centers! Not applicable in Hookium.
			vpq = gbf.Vpq(bpos[p], bpos[q], jnp.array([0.0, 0.0, 0.0]), bparam[p], bparam[q], k) # VCPQ TODO Z=k at [0.0,0.0,0.0] atm
			V = jax.ops.index_update(V, (p, q), vpq)
			V = jax.ops.index_update(V, (q, p), vpq)

	return V

def SCFLoop(bpos, bparam, M, nel, maxiter, mintol, k, D0=None, C0=None):	
	"""
	The self consistent field loop is the procedure of finding the density matrix with the lowest
	energy. It consists of steps
	i) D -> F step, where the Fock matrix is constructed from existing density
	ii) F -> D step, where the new density is obtained from Fock matrix by solving
		the generalized eigenvalue problem:

					FC = SCA

		giving C, from which D is constructed. 
	
	"""

	# construct overlap matrix
	S = constructS(bpos, bparam, M)
	T = constructT(bpos, bparam, M)
	V = constructV(bpos, bparam, M, k) # TODO Fix for System
	V2B = constructV2B(bpos, bparam, M)

	if D0 is None:
		if C0 is None:
			C0 = initC(M, nel)	
		D = constructD0(C0, M, nel)

	E0 = jnp.inf
	i = 0
	while i < maxiter:
		i += 1
		G = constructG(D, V2B, bpos, bparam, M)
		F = T+V+G

		### F -> D step
		C = solveGEP(F, S)
 
		### D -> F step
		D = constructD(C, M, nel)

		# Check energy
		E = SCFEnergy(D, T, V, G)

		# TODO add nuclear interaction

		print("After iteration {}: E = {}".format(i, E))
		if(jnp.abs(E-E0) < mintol):
			break

		E0 = E

	return E, D

# @jit
def solveGEP(F,S):
	"""
	Solve the generalized eigenvalue problem in such a way that jax can differentiate through it!
	
	TODO figure out how many iterations are needed.

	"""
	w, V = eigh.eigh(F, S)
	# w, V = scipy.linalg.eigh(F, b=S)


	return V


@jit
def SCFEnergy(D, T, V, G):
	return jnp.trace(jnp.dot(D, (T+V))) + 0.5*jnp.trace(jnp.dot(D, G))