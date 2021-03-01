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
import numpy as np
import Utilities.Wavefunction.GaussianBasis as basis

def constructF(D, basis, M):
	"""
	Construct the fock matrix
	"""
	F = jnp.zeros((M, M), dtype=np.float32)
	
	for p in range(M):
		for q in range(M):

			# Calculate single fpq element of matrix
			gpq = 0.0

			# Two-electron integrals
			for r in range(M):
				for s in range(M):
					gpq += D[r, s]*(basis.pqrs()-0.5*basis.pqrs())

			# Kinetic term
			tpq = basis.Tpq()

			# Kinetic term
			vpq = basis.Vpq()

			# fpq is sum of kinetic, potential and two-el. term
			fpq = gpq + tpq + vpq

			# update F
			F = jax.ops.index_update(F, (p, q), fpq)

	return F

def constructS(basis, M):
	"""
	Construct the overlap matrix
	"""
	S = jnp.zeros((M, M), dtype=np.float32)

	for p in range(M):
		for q in range(M):
			spq = basis.Spq()
			S = jax.ops.index_update(S, (p, q), spq)

	return S

def constructD(C, basis, M):
	"""
	Construct density matrix
	"""
	D = jnp.zeros((M, M), dtype=np.float32)

	for p in range(M):
		for q in range(M):
			dpq = 0.0
			for a in range(M//2): # todo, doublecheck if this is okay!
				dpq += C[a, p]*C[a, q]

			D = jax.ops.index_update(D, (p, q), 2.0*dpq)

	return D
	
def getElDensity(r, D, basis, M):
	rho = 0.0
	for p in range(M):
		for q in range(M):
			rho += D[p, q]*basis.eval()*basis.eval()
	return rho

def SCFLoop(C0, basis, maxiter, mintol):
	"""
	Self consistent loop. 
	"""
