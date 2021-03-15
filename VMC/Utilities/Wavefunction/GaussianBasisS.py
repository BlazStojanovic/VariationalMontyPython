"""	
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 24/02/2021
----------
Contains: 
	Gaussian basis integrals, but only with s-type orbitals. This is because stype orbitals are sufficient for the
	description of Hookium and the evaluation of specialized function for stype orbitals is much quicker than
	the general solutions in GaussianBasis.py and GaussianBasisGeneral.py. Moreover the functions can be jitted as there is no recursion
	involved. 

	Integral solutions are from Modern Quantum Chemistry by Szabo and Ostlund
"""

import jax.numpy as jnp
import jax.scipy.special as special
from jax import jit, vmap
import jax.lax as lax

@jit
def normFactor(alpha):
	"""
	Evaluate the normalization constant of an s-type Gaussian orbital. (1 dimensional)
	"""
	return jnp.power(2*alpha/jnp.pi, 0.75)

@jit
def K(r1, r2, p1, p2):
	"""
	Evaluate the basic integral of a s-type gaussian orbital. See Szabo and Ostlund, pg. 410
	"""
	a1 = p1[0]
	a2 = p2[0]
	p = a1 + a2
	return normFactor(a1)*normFactor(a2)*jnp.exp(-a1*a2*di(r1, r2)/p)

@jit
def di(r1, r2):
	"""
	Evaluate distance between r1 and r2
	"""
	# Avoid NaNs
	d = r1-r2
	is_zero = jnp.allclose(d, 0.)
	d = jnp.where(is_zero, jnp.ones_like(d), d)  # replace d with ones if is_zero
	l = jnp.linalg.norm(d)
	l = jnp.where(is_zero, 0., l)  # replace norm with zero if is_zero

	return jnp.square(l)

@jit
def GPC(r1, r2, p1, p2):
	"""
	Evaluate Gaussian product centre, i.e. the point at which the Gaussian that is a product
	of two Gaussians is centerd.
	"""
	a1 = p1[0]
	a2 = p2[0]
	return (a1*r1+a2*r2)/(a1+a2)

@jit
def evaluate(r, rc, params):
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
	"""

	# alpha = params[0]
	alpha = params.T
	R = r - rc

	return normFactor(alpha)*jnp.exp(-alpha*jnp.square(jnp.linalg.norm(R, axis=-1)))

@jit
def Spq(r1, r2, p1, p2):
	"""
	Overlap matrix element for two cartesian Gaussians centered at r1 and r2, 
	with distinct coefficients alpha and ijk
	
	Parameters
	----------
	r1: jnp.array of shape (1, 3)
		position of the	first Gaussian,
	 	assumed to be (0,0,0) but left as parameter to fit general structure of HF module
	
	r2: jnp.array of shape (1, 3)
		position of the	second Gaussian,
		assumed to be (0,0,0) but left as parameter to fit general structure of HF module
	
	p1: jnp.array of shape (1, 1)
		parameters of the first gaussian

	p2: jnp.array of shape (1, 1)
		parameters of the second gaussian

	Returns
	----------
	Spq for the two basis functions

	"""
	a1 = p1[0]
	a2 = p2[0]
	p = a1 + a2

	return jnp.power((jnp.pi/p), 1.5)*K(r1, r2, p1, p2)

@jit
def Tpq(r1, r2, p1, p2):
	"""
	Kinetic energy contribution of two primitive Gaussians

	Parameters
	----------
	r1, r2: (1, 3) center positions of the primitives
	p1, p2: parameters of each Gaussian,containing alpha

	Returns
	---------
	Tpq matrix element (float)

	"""
	a1 = p1[0]
	a2 = p2[0]
	p = a1 + a2
	q = a1*a2/p

	return q*(3.0-2.0*q*di(r1, r2))*jnp.power((jnp.pi/p), 1.5)*K(r1, r2, p1, p2)

@jit
def Vx(x1, x2, xc, p1, p2):
	"""
	One dimensional contribution to potential integration.

	"""
	a1 = p1[0]
	a2 = p2[0]

	enu = (a1+a2+2*a1**2*(x1-xc)**2+4*a1*a2*(x1-xc)*(x2-xc)+2*a2*(x2-xc)**2)*jnp.sqrt(jnp.pi)*jnp.exp(-a1*a2*(x1-x2)**2/(a1+a2))
	den = 2*jnp.power(a1+a2, 2.5)

	return enu/den

@jit
def Sx(x1, x2, p1, p2):
	"""
	One dimensional overlap integral.
	"""
	a1 = p1[0]
	a2 = p2[0]
	p = a1 + a2

	return jnp.sqrt(jnp.pi/p)*jnp.exp(-a1*a2*(x1-x2)**2/(a1+a2))

@jit
def Vpq(r1, r2, rc, p1, p2, k):
	"""
	Potential energy contribution of two primitive Gaussians, 
	in harmonic potential, used for Harmonium type atomic centers. 

	Parameters
	----------
	r1, r2: (1, 3) Center of orbital 
	rc: (1, 3) Center of nuclear
	p1, p2: parameters of each Gaussian,containing alpha
	k: Spring coefficient

	Returns
	---------
	Vpq matrix element (float)

	"""

	a1 = p1[0]
	a2 = p2[0]

	Vpqx = Vx(r1[0], r2[0], rc[0], p1, p2)*Sx(r1[2], r2[2], p1, p2)*Sx(r1[1], r2[1], p1, p2)
	Vpqy = Vx(r1[1], r2[1], rc[1], p1, p2)*Sx(r1[2], r2[2], p1, p2)*Sx(r1[0], r2[0], p1, p2)
	Vpqz = Vx(r1[2], r2[2], rc[2], p1, p2)*Sx(r1[1], r2[1], p1, p2)*Sx(r1[0], r2[0], p1, p2)

	N = normFactor(a1)*normFactor(a2)

	return k*0.5*(Vpqx+Vpqy+Vpqz)*N

@jit
def Vcpq(r1, r2, rc, p1, p2, Z):
	"""
	Potential energy contribution of two primitive Gaussians (NUCLEAR!!)

	Parameters
	----------
	r1, r2: (1, 3) Center of orbital 
	rc: (1, 3) Center of nuclear
	p1, p2: parameters of each Gaussian,containing alpha
	Z: charge

	Returns
	---------
	VCpq matrix element (float)

	"""
	a1 = p1[0] 
	a2 = p2[0]
	p = a1 + a2
	t = p*jnp.power(di(GPC(r1, r2, p1, p2), rc), 2)

	return (-2.0*jnp.pi*Z/p)*K(r1, r2, p1, p2)*boys(t)

@jit
def boys(t):
	"""
	Boys function of zeroth order (which is applicable to s-type Gaussian orbitals)
	
	Parameters
	----------
	t, float at which to evaluate the Boys function. 
	"""
	T = lambda x: 1.0
	F = lambda x: jnp.power(1.0/2.0*(jnp.pi/x), 0.5)*special.erf(jnp.power(x, 0.5))
	return lax.cond(t==0, T, F, operand=t)

@jit
def pqrs(r1, r2, r3, r4, p1, p2, p3, p4):
	"""
	Calculates the two-body electron interaction term, 
	Gaussian s-type orbitals.

	Parameters
	----------
	p1, p2, p3, p4: parameters of each gaussian each containing l,m,n and alpha
	r1, r2, r3, r4: are (1, 3) coordinates of gaussian centres

	Returns
	----------
	(pq|rs) matrix element (float)
	
	"""

	a1 = p1[0]
	a2 = p2[0]
	a3 = p3[0]
	a4 = p4[0]

	pa = a1 + a2
	pb = a3 + a4

	rpa = GPC(r1, r2, p1, p2)
	rpb = GPC(r3, r4, p3, p4)
	d = di(rpa, rpb)
	t = pa*pb/(pa+pb)*d

	N = 2.0*jnp.power(jnp.pi, 2.5)/(pa*pb*jnp.sqrt(pa+pb))

	return N*boys(t)*K(r1, r2, p1, p2)*K(r3, r4, p3, p4)