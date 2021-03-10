"""	
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 24/02/2021
----------
Contains: 
	Helper function for defining, evaluating and optimizing Gaussian expansions of single particle wf's.

	TODO proper credit assignment.

	The code in this file is only slightly adapted from the code of Joshua Goings which draws from Helgaker, 
	Trygve, and Peter R. Taylor. “Gaussian basis sets and molecular integrals.” 
	Modern Electronic Structure (1995).
"""

import jax.numpy as jnp
from jax import jit

from scipy.special import hyp1f1

def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)

def Boys(n, T):
	"""
	Boys function, express it with gammaln and gammaincc as 

	1/2 x^(-(1/2) - n) (Gamma[1/2 + n] - Gamma[1/2 + n, x])

	"""

	pass


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

def E(i,j,t,d,alpha1,alpha2):
	""" 
	Recursive definition of Hermite Gaussian coefficients.

	Parameters
	----------
	alpha1: orbital exponent on first Gaussian
	alpha2: orbital exponent on second Gaussian
	i,j: orbital angular momentum number on first and second Gaussian
	t: number nodes in Hermite (depends on type of integral, e.g. always zero for overlap integrals)
	    
	d: distance between first and second gaussian centres

	Returns
	----------	
	Hermite Gaussian coefficients.

	"""
	p = alpha1+alpha2
	q = alpha1*alpha2/p
	if (t < 0) or (t > (i + j)):
		# out of bounds for t  
		return 0.0
	elif i == j == t == 0:
		# base case
		return jnp.exp(-q*d*d) # K_AB
	elif j == 0:
		# decrement index i
		return (1/(2*p))*E(i-1,j,t-1,d,alpha1,alpha2) - (q*d/alpha1)*E(i-1,j,t,d,alpha1,alpha2) + (t+1)*E(i-1,j,t+1,d,alpha1,alpha2)
	else:
		# decrement index j
		return (1/(2*p))*E(i,j-1,t-1,d,alpha1,alpha2) + (q*d/alpha2)*E(i,j-1,t,d,alpha1,alpha2) + (t+1)*E(i,j-1,t+1,d,alpha1,alpha2)

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
	
	alpha1,l1,m1,n1 = p1 # shell angular momentum on first Gaussian
	alpha2,l2,m2,n2 = p2 # shell angular momentum on second Gaussian

	S1 = E(l1,l2,0,r1[0]-r2[0],alpha1,alpha2) # X
	S2 = E(m1,m2,0,r1[1]-r2[1],alpha1,alpha2) # Y
	S3 = E(n1,n2,0,r1[2]-r2[2],alpha1,alpha2) # Z
	return normFactor(l1,m1,n1,alpha1)*normFactor(l2,m2,n2,alpha2)*S1*S2*S3*jnp.power(jnp.pi/(alpha1+alpha2),1.5) # TODO doublecheck prefactor

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

	alpha1,l1,m1,n1 = p1
	alpha2,l2,m2,n2 = p2
	alpha3,l3,m3,n3 = p3
	alpha4,l4,m4,n4 = p4
	N1 = normFactor(l1, m1, n1, alpha1)
	N2 = normFactor(l2, m2, n2, alpha2)
	N3 = normFactor(l3, m3, n3, alpha3)
	N4 = normFactor(l4, m4, n4, alpha4)
	N = N1*N2*N3*N4

	p = alpha1+alpha2 # composite exponent for P (from Gaussians 1 and 2)
	q = alpha3+alpha4 # composite exponent for Q (from Gaussians 3 and 4)
	alpha = p*q/(p+q)
	P = gaussian_product_center(alpha1,r1,alpha2,r2) # r1 and B composite center
	Q = gaussian_product_center(alpha3,r3,alpha4,r4) # C and D composite center
	RPQ = np.linalg.norm(P-Q)

	val = 0.0
	for t in range(l1+l2+1):
		for u in range(m1+m2+1):
			for v in range(n1+n2+1):
				for tau in range(l3+l4+1):
					for nu in range(m3+m4+1):
						for phi in range(n3+n4+1):
							val += E(l1,l2,t,r1[0]-r2[0],alpha1,alpha2) * \
							E(m1,m2,u,r1[1]-r2[1],alpha1,alpha2) * \
							E(n1,n2,v,r1[2]-r2[2],alpha1,alpha2) * \
							E(l3,l4,tau,r3[0]-r4[0],alpha3,alpha4) * \
							E(m3,m4,nu ,r3[1]-r4[1],alpha3,alpha4) * \
							E(n3,n4,phi,r3[2]-r4[2],alpha3,alpha4) * \
							np.power(-1,tau+nu+phi) * \
							R(t+tau,u+nu,v+phi,0,\
							   alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)

							val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
	return N*val

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

    alpha1,l1,m1,n1 = p1
    alpha2,l2,m2,n2 = p2
    N1 = normFactor(l1, m1, n1, alpha1)
    N2 = normFactor(l2, m2, n2, alpha2)
    
    term0 = alpha2*(2*(l2+m2+n2)+3)*Spq(r1, r2, p1, p2)
    
    term1 = 0.0
    p2t = p2 + jnp.array([0, 2, 0, 0])
    term1 += Spq(r1, r2, p1, p2t)
    p2t = p2 + jnp.array([0, 0, 2, 0])
    term1 += Spq(r1, r2, p1, p2t)
    p2t = p2 + jnp.array([0, 0, 0, 2])
    term1 += Spq(r1, r2, p1, p2t)
    term1 *= -2*np.power(b,2)

    term2 = 0.0
    p2t = p2 + jnp.array([0, -2, 0, 0])
    term2 = l2*(l2-1)*Spq(r1, r2, p1, p2t)
    p2t = p2 + jnp.array([0, 0, -2, 0])
    term2 = m2*(m2-1)*Spq(r1, r2, p1, p2t)
	p2t = p2 + jnp.array([0, 0, 0, -2])
    term2 = n2*(n2-1)*Spq(r1, r2, p1, p2t)    

    term2 *= -0.5
    return N1*N2*(term0+term1+term2)

def Vpq():

	pass



if __name__ == '__main__':
	pass