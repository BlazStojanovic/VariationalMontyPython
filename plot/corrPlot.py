import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

Ehfs = np.load("../data/vmcEnergies/corr-Ehf.npy")
Evs = np.load("../data/vmcEnergies/corr-Ev.npy")
Sigvs = np.load("../data/vmcEnergies/corr-sigma.npy")

N = len(Ehfs)
k = np.linspace(0.1, 2, N, endpoint=True)

print(N)

fig, ax1 = plt.subplots()
fig.set_size_inches((12, 7))
ax2 = ax1.twinx()

ax1.plot(k, Ehfs, 'r-', label='HF energy')
ax1.plot(k, Evs, 'b-', label='VMC energy')
ax1.plot(k, Evs+Sigvs, 'b-', alpha=0.7)
ax1.plot(k, Evs-Sigvs, 'b-', alpha=0.7)
ax1.fill_between(k, Evs+Sigvs, y2=Evs-Sigvs, color='b', alpha=0.3)

ax2.plot(k, Ehfs-Evs, 'k--', label='Correlation')


ax1.legend(loc='center right')
ax2.legend(loc='upper right')
ax1.set_xlabel("$k$")
ax1.set_ylabel("$E$ [a.u.]")
ax1.set_ylabel(r"$E_c$ [a.u.]")

plt.savefig('../plots/corrPlot.png', bbox_inches='tight')
plt.show()