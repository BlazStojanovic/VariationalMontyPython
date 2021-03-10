import Utilities
from Utilities.HartreeFock import HartreeFockG as hf
from Utilities.Wavefunction import System
from Utilities.Wavefunction import GaussianBasisS as gbfs

import jax
import jax.numpy as jnp
from jax import grad
import numpy as np
 
if __name__ == '__main__':


	## STO-3G
	M = 3
	nel = 2

	bpos = jnp.zeros((M, 3))
	bparam = jnp.array([[0.6362421394E+01],
					    [0.1158922999E+01],
					    [0.3136497915E+00]])
	C = jnp.array([[0.9163596281E-02, 0.4936149294E-01, 0.1685383049E+00]])


	# Norm factor derivative
	nfd = grad(gbfs.normFactor)
	print(nfd(2.0))

	# K derivative
	kd = grad(gbfs.K)	
	r1 = jnp.array([0.2, 0.56, 0.01])
	r2 = jnp.array([-0.4, 2.3, 8.0])
	p1 = jnp.array([1.2])
	p2 = jnp.array([1.9])
	print(kd(r1, r2, p1, p2))