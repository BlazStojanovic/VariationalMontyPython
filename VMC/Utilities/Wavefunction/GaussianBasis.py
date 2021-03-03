"""	
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 24/02/2021
----------
Contains: 
	Basic Gaussian orbital set where only one site at r = (0, 0, 0) is allowed. This massively simplifies the
	evaluations of relevant integrals. The general basis set can be found in GaussianBasisGeneral.py.
"""

import jax.numpy as jnp
from jax import jit

def eval(r, rc, params):
	"""
	A single gaussian s-orbital like basis function. 

	Parameters
    ----------
    r: jnp.array
       position to be evaluated at, shape = (1, 3)

	rc: jnp.array
	    position at which the basis function is centered, shape = (1, 3)

    params: array of parameters which determine the shape of the basis function, 
    		params[0] = alpha: width of the gaussian
    		params[1] = i: order of the x bf.
    		params[2] = j: order of the x bf.
    		params[3] = k: order of the x bf.
	"""

	alpha, i, j, k = params

	R = r - rc
	
	return normFactor(i,j,k,alpha)*\
		   jnp.power(R[0],i)*\
		   jnp.power(R[1],j)*\
		   jnp.power(R[2],k)*\
		   jnp.exp(-alpha*jnp.power(jnp.linalg.norm(R), 2))

def Spq(r1, r2, p1, p2):
	"""
	Overlap matrix element for two cartesian Gaussians centered at r1 and r2, 
	with distinct coefficients alpha and ijk
	
	Parameters
	----------
	r1: jnp.array of shape (1, 3)
		position of the	first Gaussian

	r2: jnp.array of shape (1, 3)
		position of the	second Gaussian

	p1: jnp.array of shape (1, 4)
		parameters of the first gaussian

	p2: jnp.array of shape (1, 4)
		parameters of the second gaussian

	Returns
	----------
	Spq for the two basis functions

	"""
	pass

def normFactor(l,m,n,alpha):
	B = jnp.power(8*alpha, l+m+n)/(inbFact(2*l,l)*inbFact(2*m,m)*inbFact(2*n,n))
	return jnp.power(2*alpha/jnp.pi, 3.0/4.0)*jnp.power(B, 0.5)

def inbFact(n, m):
	"""
	Calculates n!/m!
	n>m
	"""
	f = 1.0
	for i in range(m+1, n+1):
		f*=i
	return f

def pqrs(r1, r2, r3, r4, p1, p2, p3, p4):
	"""
	Calculates the two-body electron interaction term

	Parameters
	----------
	p1, p2, p3, p4: parameters of each gaussian each containing l,m,n and alpha
	r1, r2, r3, r4: are (1, 3) coordinates of gaussian centres

	Returns
	----------
	(pq|rs) matrix element (float)
	
	"""

	pass

def Tpq(r1, r2, p1, p2):
	"""
	Kinetic energy contribution of two primitive Gaussians

	Parameters
	----------
	r1, r2: (1, 3) center positions of the primitives
	p1, p2: parameters of each Gaussian, containing i,j,k and alpha

	Returns
	---------
	Tpq matrix element (float)

	"""
	pass

def Vpq():
	pass

def R(t,u,v,n,p,PCx,PCy,PCz,RPC):
    ''' Returns the Coulomb auxiliary Hermite integrals 
        Returns a float.
        Arguments:
        t,u,v:   order of Coulomb Hermite derivative in x,y,z
                 (see defs in Helgaker and Taylor)
        n:       order of Boys function 
        PCx,y,z: Cartesian vector distance between Gaussian 
                 composite center P and nuclear center C
        RPC:     Distance between P and C
    '''
    T = p*RPC*RPC
    val = 0.0
    if t == u == v == 0:
        val += np.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,PCx,PCy,PCz,RPC)
        val += PCz*R(t,u,v-1,n+1,p,PCx,PCy,PCz,RPC)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCy*R(t,u-1,v,n+1,p,PCx,PCy,PCz,RPC)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,PCx,PCy,PCz,RPC)
        val += PCx*R(t-1,u,v,n+1,p,PCx,PCy,PCz,RPC)
    return val

if __name__ == '__main__':
	pass