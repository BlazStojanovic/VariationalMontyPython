"""	
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 12/03/2021
----------
Contains: 
	All necessary helper functions to do with the Jastrow factor. 
	They are constructed with automatic differentiation in mind.
"""

import jax
import jax.numpy as jnp

def uB(rij, a, b):
	return a*rij*jnp.reciprocal(1 + b*rij)

def uA(rij, a, b):
	"""
	a and b are lists!
	"""
	return jnp.sum() # TODO

def JastrowB(rij, a, b):
	return jnp.exp(uB(rij, a, b))

