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



if __name__ == '__main__':

	## STO-3G
	M = 3
	nel = 2
	# bpos = jnp.zeros((M, 3))
	
	# bparam = jnp.array([[0.6362421394E+01],
	# 				    [0.1158922999E+01],
	# 				    [0.3136497915E+00]])
	# C = jnp.array([[0.9163596281E-02, 0.4936149294E-01, 0.1685383049E+00]])

	# M = 6
	# bparam = jnp.array([[0.6598456824E+02],
	# 				    [0.1209819836E+02],
	# 				    [0.3384639924E+01], 
	# 				    [0.1162715163E+01],
	# 				    [0.4515163224E+00],
	# 				    [0.1859593559E+00]])
	# C = jnp.array([[0.9163596281E-02, 0.4936149294E-01, 0.1685383049E+00, 0.3705627997E+00, 0.4164915298E+00, 0.1303340841E+00]])

	M = 8
	bparam = jnp.array([[2.4],
					    [1.2],
					    [0.6],
					    [0.37297],
					    [0.30241],
					    [0.23185],
					    [0.0750],
					    [0.0375]])
	# C = jnp.array([[0.9163596281E-02, 0.4936149294E-01, 0.1685383049E+00]])
	# C = jnp.array()

	# M=10
	# bparam = jnp.array([[4.785000E+03],
	# 					[7.170000E+02],
	# 					[1.632000E+02],
	# 					[4.626000E+01],
	# 					[1.510000E+01],
	# 					[5.437000E+00],
	# 					[2.088000E+00],
	# 					[8.297000E-01],
	# 					[3.366000E-01],
	# 					[1.369000E-01]])

	# M = 3
	# nel = 2
	bpos = jnp.zeros((M, 3))
	# bparam = orbitals.sto3g_exponents.get('He-s')
	# C = orbitals.sto3g_coefficients('He-s')

	hookium = System.System(ccoefs=[0.25]) # default init is hookium
	# hookium = System.System(ccoefs=[2.0]) # default init is hookium

	# # Density in the z=0 plane
	# x = jnp.linspace(-5, 5, 100)
	# y = jnp.linspace(-5, 5, 100)
	# Z = jnp.zeros((1, 100*100))

	# X, Y = jnp.meshgrid(x, y)
	# X = jnp.ravel(X)
	# Y = jnp.ravel(Y)

	# R = jnp.vstack((X, Y, Z)).T

	# print("R", R)

	# Rc = jnp.array([0, 0, 0], dtype=jnp.float32)

	# rho = hf.getElDensity(R, D, bpos, bparam, M)
	# print(jnp.shape(rho))
	# rho = jnp.reshape(rho, (100, 100))
	# X = jnp.reshape(X, (100, 100))
	# Y = jnp.reshape(Y, (100, 100))
	# print(jnp.shape(rho))

	# import matplotlib.pyplot as plt
	# plt.pcolormesh(X, Y, rho)
	# plt.colorbar()
	# plt.show()
	
	print(bparam)
	maxiter = 100
	mintol  = 1e-5
	E, D = hf.SCFLoop(bpos, bparam, M, nel, maxiter, mintol, hookium.ccoefs[0])

	# gradient = grad(hf.SCFLoop, argnums=2)
	# print(gradient(C, bpos, bparam, M, nel, maxiter, mintol, hookium.ccoefs[0]))

	# print(gradient(jnp.array([0.0,0.0,0.0]), jnp.array([0.0,0.0,0.0]), jnp.array([0.6362421394E+01, 0, 0, 0]), jnp.array([0.1158922999E+01, 0, 0, 0])))
	# g = grad(lambda x: jnp.sin(x))
	# print(g(2.0))