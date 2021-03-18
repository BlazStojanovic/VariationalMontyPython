"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 23/02/2021
----------
Contains: 
	Markov Chain Monte carlo code, 
	helper functions for Metropolis Sampling.
"""
import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import jit, grad, vmap, partial

@jit
def trialProb(config, nconfig, drift, tau, N=2):
	"""
	Trial probability for moving from config to nconfig
	
	Parameters:
	----------
	config: current configuration

	nconfig: proposed configuration

	tau: timestep

	Returns:
	----------
	T(config -> nconfig)

	"""
	n = jnp.power(2.0*jnp.pi*tau, 1.5*N)
	# TODO this may need fixing if gradients are needed
	return jnp.exp(-jnp.square(jnp.linalg.norm(nconfig-config-tau*drift))/(2.0*tau))/n

@jit
def greenMove(config, drift, tau, prngkey):
	"""
	Proposes a trial move drawn from a isotropic gaussian with 
	a given average step size. 

	Parameters:
	----------
	config: current configuration of the system

	drift: the drift velocity, setting it to zero gives gaussian trial move probability

	tau: timestep

	Returns:
	----------
	new configuration
	"""
	# Perturbation drawn from Normal distribution with average of zero and standard deviation of sqrt of timestep
	return config + rnd.normal(prngkey, jnp.shape(config), dtype=jnp.float32)*jnp.sqrt(tau) + drift*tau

