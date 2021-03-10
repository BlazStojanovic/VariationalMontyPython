"""	
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 24/02/2021
----------
Contains: 
	Basic Gaussian orbital set where only one site at r = (0, 0, 0) is allowed. This massively simplifies the
	evaluations of relevant integrals. The general basis set can be found in GaussianBasisGeneral.py.

	TODO proper credit assignment.

	The code in this file is only slightly adapted from the code of Joshua Goings which draws from Helgaker, 
	Trygve, and Peter R. Taylor. “Gaussian basis sets and molecular integrals.” 
	Modern Electronic Structure (1995).
"""

import jax.numpy as jnp
import jax.scipy.special as special
from jax import jit, vmap
import jax.lax as lax

@jit
def normFactor(l,m,n,alpha):
	temp = jnp.power(8*alpha, l+m+n)*fact(l)*fact(m)*fact(n)\
									/fact(2*l)/fact(2*m)/fact(2*n)
	return jnp.power(2*alpha/jnp.pi, 3.0/4.0)*jnp.power(temp, 0.5)

@jit # Todo fix
def fact(n):
	return jnp.exp(special.gammaln(n+1))


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
    		params[1] = i: order of the x bf.
    		params[2] = j: order of the x bf.
    		params[3] = k: order of the x bf.
	"""

	alpha, i, j, k = params

	R = r - rc
	
	return normFactor(i,j,k,alpha)*\
		   jnp.power(R[:, 0],i)*\
		   jnp.power(R[:, 1],j)*\
		   jnp.power(R[:, 2],k)*\
		   jnp.exp(-alpha*jnp.power(jnp.linalg.norm(R, axis=-1), 2))

# @jit
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

# @jit
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
	
	p1: jnp.array of shape (1, 4)
		parameters of the first gaussian

	p2: jnp.array of shape (1, 4)
		parameters of the second gaussian

	Returns
	----------
	Spq for the two basis functions

	"""
	
	a1, i, j, k = p1
	a2, l, m, n = p2

	S1 =((i+l+1) % 2)\
		*jnp.sqrt(jnp.power(a1+a2, -(1.0+i+l)))\
		*jnp.exp(special.gammaln((i+l+1.0)*0.5))
	
	S2 = ((j+m+1) % 2)\
		*jnp.sqrt(jnp.power(a1+a2, -(1.0+j+m)))\
		*jnp.exp(special.gammaln((j+m+1.0)*0.5))
	
	S3 = ((k+n+1) % 2)\
		*jnp.sqrt(jnp.power(a1+a2, -(1.0+k+n)))\
		*jnp.exp(special.gammaln((k+n+1.0)*0.5))

	# N1 = normFactor(i, j, k, a1)
	# N2 = normFactor(l, m, n, a2)
	# Norm = N1*N2 # TODO fix norm

	# S1 = E(i,l,0,0,a1,a2) # X
	# S2 = E(j,m,0,0,a1,a2) # Y
	# S3 = E(k,n,0,0,a1,a2) # Z

	return S1*S2*S3 #*jnp.power(jnp.pi/(a1+a2),1.5)

# @jit
def Vpq(r1, r2, p1, p2):
	"""	
	Nuclear attraction matrix element for two cartesian Gaussians centered at r1 and r2, 
	with distinct coefficients alpha and ijk
	
	Parameters
	----------
	r1: jnp.array of shape (1, 3)
		position of the	first Gaussian,
	 	assumed to be (0,0,0) but left as parameter to fit general structure of HF module
	
	r2: jnp.array of shape (1, 3)
		position of the	second Gaussian,
		assumed to be (0,0,0) but left as parameter to fit general structure of HF module
	
	p1: jnp.array of shape (1, 4)
		parameters of the first gaussian

	p2: jnp.array of shape (1, 4)
		parameters of the second gaussian

	Returns
	----------
	scalar Vpq for the two basis functions

	"""
	Vx = Spq(r1, r2, p1, p2 + jnp.array([0, 2, 0, 0])) 
	Vy = Spq(r1, r2, p1, p2 + jnp.array([0, 0, 2, 0]))
	Vz = Spq(r1, r2, p1, p2 + jnp.array([0, 0, 0, 2]))

	alpha1,l1,m1,n1 = p1	
	alpha2,l2,m2,n2 = p2
	N1 = normFactor(l1, m1, n1, alpha1)
	N2 = normFactor(l2, m2, n2, alpha2)
	N = N1*N2
	return N*(Vx+Vy+Vz)

# @jit
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
	N = N1*N2
	term0 = alpha2*(2*(l2+m2+n2)+3)*Spq(r1, r2, p1, p2)

	term1 = 0.0
	p2t = p2 + jnp.array([0, 2, 0, 0])
	term1 += Spq(r1, r2, p1, p2t)
	p2t = p2 + jnp.array([0, 0, 2, 0])
	term1 += Spq(r1, r2, p1, p2t)
	p2t = p2 + jnp.array([0, 0, 0, 2])
	term1 += Spq(r1, r2, p1, p2t)
	term1 *= -2*jnp.power(alpha2,2)

	term2 = 0.0
	p2t = p2 + jnp.array([0, -2, 0, 0])
	term2 = l2*(l2-1)*Spq(r1, r2, p1, p2t)
	p2t = p2 + jnp.array([0, 0, -2, 0])
	term2 = m2*(m2-1)*Spq(r1, r2, p1, p2t)
	p2t = p2 + jnp.array([0, 0, 0, -2])
	term2 = n2*(n2-1)*Spq(r1, r2, p1, p2t)    

	term2 *= -0.5
	return N*(term0+term1+term2)

def Vcpq(r1, r2, p1, p2):
	''' Evaluates kinetic energy integral between two Gaussians
	 Returns a float.
	 a:    orbital exponent on Gaussian 'a' (e.g. alpha in the text)
	 b:    orbital exponent on Gaussian 'b' (e.g. beta in the text)
	 lmn1: int tuple containing orbital angular momentum (e.g. (1,0,0))
	       for Gaussian 'a'
	 lmn2: int tuple containing orbital angular momentum for Gaussian 'b'
	 A:    list containing origin of Gaussian 'a', e.g. [1.0, 2.0, 0.0]
	 B:    list containing origin of Gaussian 'b'
	 C:    list containing origin of nuclear center 'C'
	'''
	a1,l1,m1,n1 = p1 
	a2,l2,m2,n2 = p2
	p = a1 + a2
	N1 = normFactor(l1, m1, n1, a1)
	N2 = normFactor(l2, m2, n2, a2)
	N = N1*N2
    # P = gaussian_product_center(a,A,b,B) # Gaussian composite center
    # RPC = np.linalg.norm(P-C)
	l1,m1,n1 = l1.astype(int), m1.astype(int), n1.astype(int)
	l2,m2,n2 = l2.astype(int), m2.astype(int), n2.astype(int)

	val = 0.0
	for t in range(l1+l2+1):
		for u in range(m1+m2+1):
			for v in range(n1+n2+1):
				val += E(l1,l2,t,0,a1,a2) * \
				E(m1,m2,u,0,a1,a2) * \
				E(n1,n2,v,0,a1,a2) * \
				R(t,u,v,0,p,0,0,0,0)
				val *= 2*jnp.pi/p 
	return N*val

# @jit
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
        val += jnp.power(-2*p,n)*boys(n,T)
    elif t == u == 0:
        if v > 1:
            val += (v-1)*R(t,u,v-2,n+1,p,0,0,0,0)
        val += PCz*R(t,u,v-1,n+1,p,0,0,0,0)
    elif t == 0:
        if u > 1:
            val += (u-1)*R(t,u-2,v,n+1,p,0,0,0,0)
        val += PCy*R(t,u-1,v,n+1,p,0,0,0,0)
    else:
        if t > 1:
            val += (t-1)*R(t-2,u,v,n+1,p,0,0,0,0)
        val += PCx*R(t-1,u,v,n+1,p,0,0,0,0)
    return val

@jit
def boys(n, T):
	"""
	Boys function for the special case when RPC = 0
	n is the order,
	T is not used, but kept for consistency.
	"""
	return 1.0/(1.0+2.0*n)

@jit
def gaussian_product_center(a,A,b,B):
    return (a*A+b*B)/(a+b)

# @jit
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

	l1,m1,n1 = l1.astype(int), m1.astype(int), n1.astype(int)
	l2,m2,n2 = l2.astype(int), m2.astype(int), n2.astype(int)
	l3,m3,n3 = l3.astype(int), m3.astype(int), n3.astype(int)
	l4,m4,n4 = l4.astype(int), m4.astype(int), n4.astype(int)


	p = alpha1+alpha2 # composite exponent for P (from Gaussians 1 and 2)
	q = alpha3+alpha4 # composite exponent for Q (from Gaussians 3 and 4)
	alpha = p*q/(p+q)
	P = gaussian_product_center(alpha1,r1,alpha2,r2) # r1 and B composite center
	Q = gaussian_product_center(alpha3,r3,alpha4,r4) # C and D composite center
	RPQ = jnp.linalg.norm(P-Q)

	val = 0.0
	for t in range(l1+l2+1):
		for u in range(m1+m2+1):
			for v in range(n1+n2+1):
				for tau in range(l3+l4+1):
					for nu in range(m3+m4+1):
						for phi in range(n3+n4+1):
							val += E(l1,l2,t,0,alpha1,alpha2) * \
							E(m1,m2,u,0,alpha1,alpha2) * \
							E(n1,n2,v,0,alpha1,alpha2) * \
							E(l3,l4,tau,0,alpha3,alpha4) * \
							E(m3,m4,nu ,0,alpha3,alpha4) * \
							E(n3,n4,phi,0,alpha3,alpha4) * \
							jnp.power(-1,tau+nu+phi) * \
							R(t+tau,u+nu,v+phi,0,\
							   alpha,P[0]-Q[0],P[1]-Q[1],P[2]-Q[2],RPQ)

							val *= 2*jnp.power(jnp.pi,2.5)/(p*q*jnp.sqrt(p+q))
	return N*val