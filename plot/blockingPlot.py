import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

def blocking_transform(E, Nb):
	Eb = np.zeros(len(E))
	n = (np.shape(E)[0])//Nb
	print(n)
	for i in range(n-1):
		Eb[i*Nb:(i+1)*Nb] = np.average(E[i*Nb:(i+1)*Nb])
	Eb[Nb*(n-1)::] = np.average(E[Nb*(n-1)::])
	return Eb

def blocking_average(E, Nb):
	ar = E[::Nb]
	n = (np.shape(E)[0])//Nb
	av = np.cumsum(ar)/(np.arange(n)+1)
	return np.repeat(av, Nb)

def blocking_var(E, Ev, Nb):
	ar = E[::Nb]
	n = (np.shape(E)[0])//Nb
	av = np.cumsum(np.square(ar-Ev))/(np.arange(n))
	return np.repeat(av, Nb)

El1 = np.load("../data/vmcEnergies/vmc-hf-1g-j1-E1.npy")
El2 = np.load("../data/vmcEnergies/vmc-hf-1g-j1-E2.npy")
Eb1 = blocking_transform(El1, 20)
Eb2 = blocking_transform(El2, 20)

bav1 = blocking_average(Eb1, 20)
bav2 = blocking_average(Eb2, 20)

Ev1 = np.average(El1)
bvar1 = blocking_var(El1, Ev1, 20)
Ev2 = np.average(El2)
bvar2 = blocking_var(El2, Ev2, 20)


fig, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
fig.set_size_inches((12, 12))
ax1.plot(El1, '-', color='k', alpha=0.5, label='$E_L$')
ax1.plot(Eb1, '-', color='red', label='$E_b$')
ax1.plot(bav1, '-', color='blue', linewidth=1, label=r'$\langle E_b \rangle$')
ax1.plot(bav1+bvar1, '-', color='blue', alpha=0.3)
ax1.plot(bav1-bvar1, '-', color='blue', alpha=0.3)
ax1.fill_between(np.arange(len(Eb1)), bav1+bvar1, bav1-bvar1, alpha=0.3, label=r'$\langle E_b \rangle \pm \sigma_b$')

ax1.set_ylim([1.5, 4])

ax2.plot(El2, '-', color='k', alpha=0.5, label='$E_L$')
ax2.plot(Eb2, '-', color='red', label='$E_b$')
ax2.plot(bav2, '-', color='blue', linewidth=1, label=r'$\langle E_b \rangle$')
ax2.plot(bav2+bvar2, '-', color='blue', alpha=0.3)
ax2.plot(bav2-bvar2, '-', color='blue', alpha=0.3)
ax2.fill_between(np.arange(len(Eb2)), bav2+bvar2, bav2-bvar2, alpha=0.3, label=r'$\langle E_b \rangle \pm \sigma_b$')

ax2.set_ylim([1.5, 4])

ax1.legend()
ax2.legend()
ax2.set_xlabel("MC time")
ax1.set_ylabel("$E$ [a.u.]")
ax2.set_ylabel("$E$ [a.u.]")
plt.savefig('../plots/blocking.png', bbox_inches='tight')
# plt.show()