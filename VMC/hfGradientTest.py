import Utilities
from Utilities.HartreeFock import HartreeFockG as hf
from Utilities.Wavefunction import System
from Utilities.Wavefunction import GaussianBasisS as gbfs
from Utilities.Wavefunction import LCGOData as orbitals

import jax
import jax.numpy as jnp
from jax import grad, jacfwd
import numpy as np

from jax.config import config
config.update("jax_debug_nans", True)
 
import matplotlib.pyplot as plt

if __name__ == '__main__':


	r1 = jnp.array([0.2, 0.56, 0.01])
	r2 = jnp.array([-0.4, 2.3, 8.0])
	p1 = jnp.array([1.2])
	p2 = jnp.array([1.9])

	# # Norm factor derivative
	# nf_g = grad(gbfs.normFactor)
	# print(nf_g(2.0))

	# # K derivative
	# k_g = grad(gbfs.K, argnums=[2])	
	# X = jnp.linspace(0, 2, 100)
	
	# for x in X:
	# 	print(x, ": ", k_g(r1, r2, [x], p2))

	# # di(r1, r2)
	# di_g = grad(gbfs.di)
	# print(di_g(r1, r2))


	########### MATRIX CONSTRUCTION TEST #############

	# nel = 2
	# bparam = orbitals.sto3g_exponents.get('He-s')
	# C = orbitals.sto3g_coefficients.get('He-s')
	# M = jnp.shape(C)[1]

	# ncs = 1
	# bpos = jnp.zeros((M, 3))
	# bpos = jnp.tile(bpos, (ncs, 1))
	# def f(bpos, bparam, M):
	# 	return jnp.sum(hf.constructS(bpos, bparam, M), axis=None)
	
	# constructS_g = grad(f, argnums=1)
	# print(constructS_g(bpos, bparam, M))

	############# THE GENERALIZED EVAL PROB TEST ###############
	# nel = 2
	# bparam = orbitals.sto3g_exponents.get('He-s')
	# C = orbitals.sto3g_coefficients.get('He-s')
	# M = jnp.shape(C)[1]

	# ncs = 1
	# bpos = jnp.zeros((M, 3))
	# bparam = jnp.tile(bparam, (ncs, 1))

	# centers = ['hooke']
	# cpos = jnp.array([[0.0, 0.0, 0.0]])
	# ccoefs = jnp.array([0.25])

	# maxiter = 1
	# mintol  = 1e-5


	# def g(bparam, cpos, ccoefs, ncs, M, nel):
	
	# 	bpos = jnp.tile(cpos, (ncs, 1))
	# 	Mt = M*ncs
	# 	S = hf.constructS(bpos, bparam, Mt)
	# 	T = hf.constructT(bpos, bparam, Mt)
	# 	V = hf.constructV(bpos, bparam, cpos[0], Mt, ccoefs[0])
	# 	V2B = hf.constructV2B(bpos, bparam, Mt)
	# 	C0 = hf.initC(Mt, nel)	
	# 	D = hf.constructD0(C0, Mt, nel)
	# 	G = hf.constructG(D, V2B, bpos, bparam, Mt)
	# 	F = T+V+G
		
	# 	# There the test begins
	# 	C = hf.solveGEP(F, S)
	# 	D = hf.constructD(C, Mt, nel)
	# 	return hf.SCFEnergy(D, T, V, G)


	# g_g = jacfwd(g, argnums=0)
	# print(g_g(bparam, cpos, ccoefs, ncs, M, nel))

	############# THE ULTIMATE TEST #################

	# HF With STO-3G basis
	nel = 2
	bparam = orbitals.sto3g_exponents.get('He-s')
	C = orbitals.sto3g_coefficients.get('He-s')
	M = jnp.shape(C)[1]

	ncs = 1
	bpos = jnp.zeros((M, 3))
	bparam = jnp.tile(bparam, (ncs, 1))

	centers = ['hooke']
	cpos = jnp.array([[0.0, 0.0, 0.0]])
	ccoefs = jnp.array([0.25])

	maxiter = 10
	mintol  = 1e-5
	# E, D = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel)

	scf_g = jacfwd(hf.SCFLoop, argnums=0)
	print(scf_g(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter))


