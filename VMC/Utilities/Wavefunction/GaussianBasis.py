"""	
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 24/02/2021
----------
Contains: 
	Helper function for defining, evaluating and optimizing Gaussian expansions of single particle wf's.

	TODO proper credit assignment.

	The conde in this file is only slightly adapted from the code of Joshua Goings which draws from Helgaker, 
	Trygve, and Peter R. Taylor. “Gaussian basis sets and molecular integrals.” 
	Modern Electronic Structure (1995).
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

def E(i,j,t,Qx,a,b):
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
		return np.exp(-q*d*d) # K_AB
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

def Tpq():
	    ''' Evaluates kinetic energy integral between two Gaussians
        Returns a float.
        a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
        b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
        lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
              for Gaussian 'a'
        lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
        A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
        B:    list containing origin of Gaussian 'b'
    '''
    l1,m1,n1 = lmn1
    l2,m2,n2 = lmn2
    term0 = b*(2*(l2+m2+n2)+3)*\
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2),B)
    term1 = -2*np.power(b,2)*\
                           (overlap(a,(l1,m1,n1),A,b,(l2+2,m2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2+2,n2),B) +
                            overlap(a,(l1,m1,n1),A,b,(l2,m2,n2+2),B))
    term2 = -0.5*(l2*(l2-1)*overlap(a,(l1,m1,n1),A,b,(l2-2,m2,n2),B) +
                  m2*(m2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2-2,n2),B) +
                  n2*(n2-1)*overlap(a,(l1,m1,n1),A,b,(l2,m2,n2-2),B))
    return term0+term1+term2

def Vpq():
	# TODO on your own tomorrow 2.3.2021
	pass

def pqrs():
def electron_repulsion(a,lmn1,A,b,lmn2,B,c,lmn3,C,d,lmn4,D):
	"""
	Calculates the two-body electron interaction term

	Parameters
	----------


	Returns
	----------
	(pq|rs) matrix element
	
	"""

	l1,m1,n1 = lmn1
	l2,m2,n2 = lmn2
	l3,m3,n3 = lmn3
	l4,m4,n4 = lmn4
	p = a+b # composite exponent for P (from Gaussians 'a' and 'b')
	q = c+d # composite exponent for Q (from Gaussians 'c' and 'd')
	alpha = p*q/(p+q)
	P = gaussian_product_center(a,A,b,B) # A and B composite center
	Q = gaussian_product_center(c,C,d,D) # C and D composite center
	RPQ = np.linalg.norm(P-Q)

	val = 0.0
	for t in range(l1+l2+1):
		for u in range(m1+m2+1):
			for v in range(n1+n2+1):
				for tau in range(l3+l4+1):
					for nu in range(m3+m4+1):
						for phi in range(n3+n4+1):
							val += E(l1,l2,t,A[0]-B[0],a,b) * \
							E(m1,m2,u,A[1]-B[1],a,b) * \
							E(n1,n2,v,A[2]-B[2],a,b) * \
							E(l3,l4,tau,C[0]-D[0],c,d) * \
							E(m3,m4,nu ,C[1]-D[1],c,d) * \
							E(n3,n4,phi,C[2]-D[2],c,d) * \
							np.power(-1,tau+nu+phi) * \
							R(t+tau,u+nu,v+phi,0,\
							   alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)

							# TODO add norm!
							val *= 2*np.power(np.pi,2.5)/(p*q*np.sqrt(p+q))
	return val


if __name__ == '__main__':
	pass