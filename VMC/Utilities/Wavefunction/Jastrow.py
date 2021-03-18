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
from jax import jit

# @jit
# def uB(rij, a, b):
# 	return a*rij*jnp.reciprocal(1.0 + b*rij)

@jit
def u(rij, a, b):
	"""
	a and b is list!
	"""
	N = jnp.shape(b)[0]
	pows = jnp.arange(N)+1
	rijs = jnp.power(jnp.ones(N)*rij, pows)
	return a*rij*jnp.reciprocal(1.0 + jnp.sum(jnp.dot(rijs, b)))

@jit
def Jastrow(rij, a, b):
	return jnp.exp(u(rij, a, b))

@jit
def Hastrow(rij, a):
	pass