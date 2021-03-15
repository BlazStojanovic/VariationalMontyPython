"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 23/02/2021
----------
Contains: 
	WaveFunction class, that will encode a wave function that describes the system, using a Basis Class
	and Jastrow factor class. This representation is automatically differentiable and is used to calculate
	energy and energy variance of the trial wf. The parameters in the LCAO expension and the Jastrow factor
	are optimizable. 
"""

from Utilities.Wavefunction import Jastrow as js

import jax
import jax.numpy as jnp
from jax import jit

def sth():
	pass

def trialWaveFunction():
	pass

