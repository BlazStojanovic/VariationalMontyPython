"""	
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 24/02/2021
----------
Contains: 
	Helper function for defining, evaluating and optimizing HO expansions of single particle wf's.
"""


# TODO, find a way to implement Hermite polynomials in JAX for optimisation!

import jax.numpy as jnp
import jax.numpy.special
from jax import jit, grad

@jit
def hermite(r, k):
	pass

@jit
def bfHO(r, k=1):
	return (-1)**k*jnp.exp(jnp.square(r))*hermite(r, k)