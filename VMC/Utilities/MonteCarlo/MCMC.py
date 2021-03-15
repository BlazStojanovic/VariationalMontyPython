"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 23/02/2021
----------
Contains: 
	Markov Chain Monte carlo code, 
	functions to perform the evaluation of local energy,
	and the Metropolis sampling. 
"""
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, partial

@partial(jit, static_argnums=0)
def Tl(twf, config, optParam, bparam, c):
	d2Psi = grad(grad(twf, argnums=0))(config, optParam, bparam, c)
	return -0.5*jnp.sum(d2Psi)

# @jit
def Vl(config, system):
	# only hookium atm, add functionallty later
	is_zero = jnp.allclose(config, 0.)
	d = jnp.where(is_zero, jnp.ones_like(config), config)  # replace d with ones if is_zero
	l = jnp.linalg.norm(config)
	l = jnp.where(is_zero, 0., l)  # replace norm with zero if is_zero

	return 0.5*k*jnp.square(l) # k/2*(r1^2 + r2^2)

# @jit
def E_local(twf, config, system):
	PsiVal = twf(config)
	return (Tl(twf, config) + Vl(config, system)*PsiVal)*jnp.reciprocal(PsiVal)

def MCMove(config, dist):
	
	dif = 0.0 # Sample from Gaussian with 

	return config + dif

