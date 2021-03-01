"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 23/02/2021
----------
Contains: 
	System class, a class for representing physical systems.
"""

import jax
import jax.numpy as jnp
from jax import jit

import GaussianBasis
import HOBasis

class System:
	
	def __init__(self, ne=2, centers=[[0.,0.,0.]], ctypes=['hooke'], ccoefs=[1/4], dim=3):
		
		# dimensionality of the problem
		self.dim = dim

		# atomic sites, centers for basis expansions
		for center in centers:
			assert len(center) == self.dim, "Positions of centers should be equal to the dimensionality of the system!"

		self.centers = centers # units in Angstroms

		# no. electrons
		assert (ne % 2 == 0) and (ne is not 0), "No. of electrons in restricted HF should be even!"
		self.ne = ne

		# center types
		for ctype in ctypes:
			assert (ctype == 'hooke'), "Only hookium center types are supported at the moment!"
		self.ctypes=ctypes

		# center coefficients (k if hookium, Z if coulomb)
		self.ccoefs = ccoefs

	def __repr__(self):
		
		s = "System: \n"
		s += "-"*20 + "\n"
		s += "Number of atomic centers: {0}".format(len(self.centers)) + "\n"
		s += "Number of electrons: {0}".format(self.ne) + "\n"
		s += "Types of atomic centers:"
		for i in range(len(self.centers)):
			s += "\n\t{0} center at {1} with coefficient k or Z = {2}".format(self.ctypes[i], self.centers[i], self.ccoefs[i])
		s += "\n" + "-"*20
		
		return s

	def addCenter(self, center, ctype, ccoef):
		assert len(center) == self.dim
		self.centers = (self.centers).append(center)
	
		self.ccoefs = (self.ccoefs).append(ccoef)

		assert (ctype == 'hooke'), "Only hookium center types are supported at the moment!"
		self.ctypes = (self.ctypes).append(ctype)

	def setNoElectrons(self, ne):
		assert (ne % 2 == 0) and (ne is not 0), "No. of electrons in restricted HF should be even!"
		self.ne = ne

	def constructBasis(self, noPerCenter, cs, coefs, basistype='Gaussian'):
		assert (basistype="Gaussian") or (basistype="HO"), "Only Gaussian and HO basis are supported!"
		self.basistype = basistype

		assert basisNo > 0
		self.basisNo = noPerCenter

		self.cs = cs
		self.coefs = coefs
		
	def outputBasis(self):
		if self.basis is None:
			print("Basis is not defined.")
		else:
			print("Basis type: {0} \n Basis per center: {1}\nExpansion coefficients: {2}\nBasis coefficients: {3}".format(self.basistype, self.basisNo, self.cs, self.basisCoefs))

if __name__ == '__main__':
	hookium = System()
	print(hookium)

	hook2 = System(ne=4, centers=[[0.,0.,0.], [0.2,0.4,0.5]], ctypes=['hooke', 'hooke'], ccoefs=[1/4, 1/2])
	print(hook2)