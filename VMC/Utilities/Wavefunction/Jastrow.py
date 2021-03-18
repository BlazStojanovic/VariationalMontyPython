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
def HeHstrow(rij, r1h, r1He, r2h, r2He, b):
	a1 = 1./2. # Opposite electrons
	a2 = 1./4. # El nucl pair
	u12 = a1*rij/(1 + rij*b[0])
	u1h = a2*r1h/(1 + r1h*b[1])
	u1He = a2*r1He/(1 + r1He*b[2])
	u2h = a2*r2h/(1 + r2h*b[3])
	u2He = a2*r2He/(1 + r2He*b[4])
	
	return jnp.exp(u12*u1h*u1He*u2h*u2He)