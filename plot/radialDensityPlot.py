"""
Plot of radial density for Hookium k=1/4.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf


import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

def radial_dens(r):
	rho = 2.0/(np.power(np.pi, 1.5)*(8.0+5.0*np.pi**0.5))
	rho *= np.exp(-0.5*np.square(r))
	rho *= np.sqrt(np.pi/2)*(7./4.+1./4.*np.square(r)+(r+np.reciprocal(r))*erf(r/np.sqrt(2))) + np.exp(-0.5*np.square(r))
	return rho*np.square(r)*2*np.pi

# analytical intracule for k=1/4
r = np.linspace(1e-12, 7, 1000)
uan = radial_dens(r)

# 1e3 samples
for i in [3, 4, 5]:
	u = np.load("../data/vmcConfigurations/vmc-1g-1e{}.npy".format(i), allow_pickle=True)
	u = np.linalg.norm(u, axis=-1)

	fig, ax = plt.subplots(figsize=[7, 7])
	ax.plot(r, uan, '--', color='blue', linewidth=3, label='Analytical')
	ax.hist(u, bins=20*i, histtype='bar', color='cornflowerblue', alpha=0.6, density=True, align='left', label='VMC')

	plt.legend()
	ax.hist(u, bins=20*i, histtype='step', color='black', density=True, align='left', label='VMC, $N={}$'.format(len(u)))
	ax.set_xlabel('$r$')
	ax.set_ylabel(r'$2\pi r^2 \rho(r)$')
	plt.savefig("../plots/rdens1e{}-1g2j.png".format(i), bbox_inches='tight')