

from Utilities.HartreeFock import HartreeFockG as hf
from Utilities.Wavefunction import LCGOData as orbitals

import jax 
import jax.numpy as jnp
from jax import jacfwd
import time

import matplotlib.pyplot as plt

def update(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter, alpha):
	db, dD, dC = jacfwd(hf.SCFLoop, argnums=0)(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-6)
	print(bparam - alpha*db)
	return bparam - alpha*db

if __name__ == '__main__':
	
	


	# Setup of initial parameters
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

	# Optimization parameters
	alpha = 0.01
	epochs = 30

	for epoch in range(epochs):
		start_time = time.time()
		bparam = update(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter, alpha)
		epoch_time = time.time() - start_time


		print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
		# E = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-6)
		# print("Self consistent Hartree-Fock Energy {}".format(E))


	E, D = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-6)	

	print("Optimal value E = {:.6f}Eh, was found with params: ".format(E), bparam)

	# Density in the z=0 plane
	x = jnp.linspace(-5, 5, 100)
	y = jnp.linspace(-5, 5, 100)
	Z = jnp.zeros((1, 100*100))

	X, Y = jnp.meshgrid(x, y)
	X = jnp.ravel(X)
	Y = jnp.ravel(Y)

	R = jnp.vstack((X, Y, Z)).T

	Rc = jnp.array([0, 0, 0], dtype=jnp.float32)

	rho = hf.getElDensity(R, D, bpos, bparam, M)
	print(jnp.shape(rho))
	rho = jnp.reshape(rho, (100, 100))
	X = jnp.reshape(X, (100, 100))
	Y = jnp.reshape(Y, (100, 100))
	print(jnp.shape(rho))

	plt.pcolormesh(X, Y, rho, shading='auto')
	plt.colorbar()
	plt.show()