"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 11/03/2021
----------
Contains: 

Estimation of Hartree-Fock limit energy of Harmonium.

"""

from Utilities.Wavefunction import LCGOData as orbitals
from Utilities.HartreeFock import HartreeFockG as hf

import numpy as np

import jax 
import jax.numpy as jnp
from jax import random

from jax import jacfwd
import time

def update(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter, alpha):
	db, (dD, dC) = jacfwd(hf.SCFLoop, argnums=0)(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-6)
	return bparam - alpha*db

if __name__ == '__main__':
	
	# exponents = [orbitals.sto2g_exponents,
	# 			 orbitals.sto3g_exponents,
	# 			 orbitals.sto4g_exponents,
	# 			 orbitals.sto5g_exponents,
	# 			 orbitals.sto6g_exponents,
	# 			 orbitals.ccpV6Z_exponents]

	# coefficients = [orbitals.sto2g_coefficients,
	# 			 orbitals.sto3g_coefficients,
	# 			 orbitals.sto4g_coefficients,
	# 			 orbitals.sto5g_coefficients,
	# 			 orbitals.sto6g_coefficients,
	# 			 orbitals.ccpV6Z_coefficients]

	exponents = [orbitals.sto3g_exponents]
	coefficients = [orbitals.sto3g_coefficients]

	# Setup of initial parameters
	nel = 2
	centers = ['hooke']
	cpos = jnp.array([[0.0, 0.0, 0.0]])
	ccoefs = jnp.array([0.25])

	maxiter = 15
	mintol  = 1e-5

	# Optimization parameters
	alpha = 0.1
	epochs = 1

	HF_energy = np.zeros((len(exponents), epochs + 1))

	i = 0
	for coef, expo in zip(coefficients, exponents):
		bparam = expo.get('He-s')
		coefficients = coef.get('He-s')
		M = jnp.shape(coefficients)[1]

		ncs = 1
		bpos = jnp.zeros((M, 3))
		bparam = jnp.tile(bparam, (ncs, 1))

		for epoch in range(epochs):
			E, (D, C) = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-8)

			HF_energy[i, epoch] = E

			print("Self consistent Hartree-Fock Energy {}".format(E))

			start_time = time.time()
			alpha *= 0.5
			bparam = update(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter, alpha)
			print(bparam)
			if bparam.any() <= 0:
				break

			epoch_time = time.time() - start_time

			print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

		E, (D, C) = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=1e-8)
		print(C)
		HF_energy[i, -1] = E	
		print("Optimal value E = {:.6f}Eh, was found with params: ".format(E), bparam)
		print(C)

		i+=1

	print(HF_energy)
	np.save("../data/hfEnergy/hf8g.npy", HF_energy)
