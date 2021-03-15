# Quick test of the HF code



import Utilities
from Utilities.HartreeFock import HartreeFockG as hf
from Utilities.Wavefunction import System
from Utilities.Wavefunction import GaussianBasisS
from Utilities.Wavefunction import LCGOData as orbitals

import jax
import jax.numpy as jnp
from jax import grad
import numpy as np

import matplotlib.pyplot as plt



if __name__ == '__main__':

	## STO-3G
	# Hookium calculation

	nel = 2
	bparam = orbitals.sto3g_exponents.get('He-s')
	C = orbitals.sto3g_coefficients.get('He-s')
	M = jnp.shape(C)[1]

	ncs = 1
	bpos = jnp.zeros((M, 3))
	bpos = jnp.tile(bpos, (ncs, 1))

	centers = ['hooke']
	cpos = jnp.array([[0.0, 0.0, 0.0]])
	ccoefs = jnp.array([0.25])

	maxiter = 100
	mintol  = 1e-5
	E, D, C = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel)
	print(E)

	# Density in the z=0 plane
	# x = jnp.linspace(-5, 5, 100)
	# y = jnp.linspace(-5, 5, 100)
	# Z = jnp.zeros((1, 100*100))

	# X, Y = jnp.meshgrid(x, y)
	# X = jnp.ravel(X)
	# Y = jnp.ravel(Y)

	# R = jnp.vstack((X, Y, Z)).T

	# Rc = jnp.array([0, 0, 0], dtype=jnp.float32)

	# rho = hf.getElDensity(R, D, bpos, bparam, M)
	# print(jnp.shape(rho))
	# rho = jnp.reshape(rho, (100, 100))
	# X = jnp.reshape(X, (100, 100))
	# Y = jnp.reshape(Y, (100, 100))
	# print(jnp.shape(rho))

	# plt.pcolormesh(X, Y, rho, shading='auto')
	# plt.colorbar()
	# plt.show()
	
	## HeH calculation

	# nel = 2
	# bparam = orbitals.sto2g_exponents.get('He-s')
	# C = orbitals.sto2g_coefficients.get('He-s')
	# M = jnp.shape(C)[1]
	# print(M)

	# ncs = 2
	# param = jnp.tile(bparam, (ncs, 1))

	# centers = ['coulomb', 'coulomb']
	# cpos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4632]])
	# ccoefs = jnp.array([2, 1])

	# E, D = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=10)

	# # Density in the z=0 plane
	# x = jnp.linspace(-5, 5, 100)
	# y = jnp.linspace(-5, 5, 100)
	# Z = jnp.zeros((1, 100*100))

	# X, Y = jnp.meshgrid(x, y)
	# X = jnp.ravel(X)
	# Y = jnp.ravel(Y)

	# R = jnp.vstack((X, Y, Z)).T

	# Rc = jnp.array([0, 0, 0], dtype=jnp.float64)
	# bpos = jnp.tile(cpos, (ncs, 1))
	# rho = hf.getElDensity(R, D, bpos, bparam, M)
	# rho = jnp.reshape(rho, (100, 100))
	# X = jnp.reshape(X, (100, 100))
	# Y = jnp.reshape(Y, (100, 100))

	# plt.pcolormesh(X, Y, rho, shading='auto')
	# plt.colorbar()
	# plt.show()
