"""
Testing VMC on Harmonium

"""
from Utilities.Wavefunction import WaveFunction as wf
from Utilities.Wavefunction import Jastrow as js
from Utilities.Wavefunction import GaussianBasisS as gbf
from Utilities.Wavefunction import LCGOData as orbitals

from Utilities.HartreeFock import HartreeFockG as hf

from Utilities.MonteCarlo import MCMC as mc

import jax
import jax.random as rnd
import jax.numpy as jnp
import jax.lax as lax
from jax import jit

# print(jax.local_device_count())

# # Define the system and Construct the basis for the trial wave funciton
# bparam = jnp.array([[6.3623724],
# 				 [1.1564877 ],
# 				 [0.21456912]]) # Optimized sto-3g

# # Solve HF problem to get density matrix and C
# nel = 2
# C = orbitals.sto3g_coefficients.get('He-s')
# # print(C)
# # print(jnp.sum(C))
# M = jnp.shape(C)[1]

# ncs = 1
# bpos = jnp.zeros((M, 3))
# bpos = jnp.tile(bpos, (ncs, 1))

# centers = ['hooke']
# cpos = jnp.array([[0.0, 0.0, 0.0]])
# ccoefs = jnp.array([0.25])

# maxiter = 15
# mintol  = 1e-5

# E, D, C = hf.SCFLoop(bparam, cpos, centers, ccoefs, ncs, M, nel, maxiter=maxiter, mintol=mintol)
# print("Hartree-Fock energy: ", E)
# print("Hartree-Fock coefficient matrix: ", C)

# # get expansion coefficients
# ci = C[:, 0]

# # Define Jastrow factor
# jastrow = js.JastrowB

# # Construct the trial wave function
# # Should have signature of (configuration, opt_par, non_opt_par)
# # @jit
# def twf(r1, r2, optParam, bparam, c):

# 	d = r1-r2
# 	is_zero = jnp.allclose(d, 0.)
# 	d = jnp.where(is_zero, jnp.ones_like(d), d)  # replace d with ones if is_zero
# 	r12 = jnp.linalg.norm(d)
# 	r12 = jnp.where(is_zero, 0., r12)  # replace norm with zero if is_zero

# 	b1 = jnp.sum(ci*gbf.evaluate(r1, jnp.zeros((1, 3)), bparam))
# 	b2 = jnp.sum(ci*gbf.evaluate(r2, jnp.zeros((1, 3)), bparam))
	
# 	a = 2 # opposite spin electron cusp condition
# 	b = optParam[0]
# 	js = jastrow(r12, a, b)

# 	return js*b1*b2


# # test of en
config = jnp.array([[1.1, 2.2, 3.3]
				   ,[-2.1, -1.2, 0.9]])

# # kin e test
# print(twf(config, [1.0], bparam, ci))
# print(mc.Tl(twf, config, [1.0], bparam, ci))

# pot test

# Perform MC integration

# Output energy, variance

# EQUILLIBRATION AND ACCUMULATION PHASE!!!!!!!!!!!

# drift = 0
# tau = 0.3
# seed = 4202
# key = rnd.PRNGKey(seed)
# for i in range(100000):
# 	key, subkey = rnd.split(key)
# 	config = mc.greenMove(config, drift, tau, subkey)


# import numpy as np

# a = np.random.random(100)

# av = 0.0
# for i in range(len(a)):
# 	av = (i*av + a[i])/(i+1)

# print(av)
# print(np.average(a))

import numpy as np
from scipy import stats
from mayavi import mlab

mu, sigma = 0, 0.1 
x = 10*np.random.normal(mu, sigma, 5000)
y = 10*np.random.normal(mu, sigma, 5000)
z = 10*np.random.normal(mu, sigma, 5000)

xyz = np.vstack([x,y,z])
kde = stats.gaussian_kde(xyz)
density = kde(xyz)

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot')
pts = mlab.points3d(x, y, z, density, scale_mode='none', scale_factor=0.07)
mlab.axes()
mlab.show()


# Plot scatter with mayavi
figure = mlab.figure('DensityPlot')
figure.scene.disable_render = True

pts = mlab.points3d(x, y, z, density, scale_mode='none', scale_factor=0.07) 
mask = pts.glyph.mask_points
mask.maximum_number_of_points = x.size
mask.on_ratio = 1
pts.glyph.mask_input_points = True

figure.scene.disable_render = False 
mlab.axes()
mlab.show()

import numpy as np
from scipy import stats
from mayavi import mlab

mu, sigma = 0, 0.1 
x = 10*np.random.normal(mu, sigma, 5000)
y = 10*np.random.normal(mu, sigma, 5000)    
z = 10*np.random.normal(mu, sigma, 5000)

xyz = np.vstack([x,y,z])
kde = stats.gaussian_kde(xyz)

# Evaluate kde on a grid
xmin, ymin, zmin = x.min(), y.min(), z.min()
xmax, ymax, zmax = x.max(), y.max(), z.max()
xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
density = kde(coords).reshape(xi.shape)

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot')

grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
min = density.min()
max=density.max()
mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))

mlab.axes()
mlab.show()