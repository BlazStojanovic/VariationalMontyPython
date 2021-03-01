"""	
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 24/02/2021
----------
Contains: 
	Helper function for defining, evaluating and optimizing Gaussian expansions of single particle wf's.
"""

import jax.numpy as jnp

def bfG(r, n=1, alpha=1):
	"""
	A single gaussian s-orbital like basis function. 

	Parameters
    ----------
    r: jnp.array
        position to be evaluated at

    n: order of basis function
	"""
	return jnp.power(r, n-1)*jnp.exp(-alpha*jnp.power(r, 2))

def psiI(r, ns, c, alphas, N):
	psi = jnp.zeros(jnp.shape(r))
	for i in range(N):
		psi += c[i]*bfG(r, n=ns[i], alpha=alphas[i])

	return psi

def Tpq():
	pass

def Vpq():
	pass

def Gpq():
	pass

def pqrs():
	pass

def Fpq():
	return 2.0

def Spq():
	pass

if __name__ == '__main__':
	r = jnp.linspace(-3, 3, 1000)
	N = 5
	ns = [1, 1, 1, 2, 2]
	c = [0.1, 0.5, 0.8, 0.8, 0.3]
	# c = [1, 0, 0, 0, 0]
	alphas = [1, 1, 1, 1, 1]


	import matplotlib.pyplot as plt
	
	plt.plot(r, psiI(r, ns, c, alphas, N))

	plt.show()