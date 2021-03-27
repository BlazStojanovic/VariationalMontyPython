"""
Plot of optimal Jastrow factors
"""

import numpy as np
import matplotlib.pyplot as plt


import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

def J(r12, b1, b2):
	return np.exp(0.5*r12/(1+b1*r12+b2*np.square(r12)))


r12 = np.linspace(0, 20, 1000)

J1 = J(r12, 0.2020724, 0.0)
J2 = J(r12, 0.22433668, -0.00510384)

figure = plt.figure(figsize=(12, 7))
plt.plot(r12, J1, 'r--', label='J1')
plt.plot(r12, J2, 'b--', label='J2')
plt.plot(r12, 1+0.5*r12, 'k-', label=r'$(1+\frac{1}{2}r_{12})$')
plt.xlabel('$r_{12}$')
plt.ylabel('$J(r_{12})$')
plt.legend()
plt.savefig('../plots/jastopt.png', bbox_inches='tight')
plt.close()
