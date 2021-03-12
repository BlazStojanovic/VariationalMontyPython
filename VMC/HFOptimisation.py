"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 11/03/2021
----------
Contains: 

This file contains an example of optimization of input parameters into a Hartree fock calculation
of Hookium, by use of forward mode jax differentiation. 

"""

from Utilities.Wavefunction import LCGOData as orbitals
from Utilities.HartreeFock import HartreeFockG as hf

import jax 
import jax.numpy as jnp
from jax import jacfwd
import time

def update(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter, alpha):
	db = jacfwd(hf.SCFLoop, argnums=0)(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-6)
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
	alpha = 0.03
	epochs = 15

	for epoch in range(epochs):
		start_time = time.time()
		bparam = update(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter, alpha)
		epoch_time = time.time() - start_time


		print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
		# E = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-6)
		# print("Self consistent Hartree-Fock Energy {}".format(E))

	# bparam = jnp.array([[6.3623724],
	# 				 [1.1564877 ],
	# 				 [0.21456912]])
	E = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-6)	

	print("Optimal value E = {:.6f}Eh, was found with params: ".format(E), bparam)