import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

Ehfs = np.load("../data/vmcEnergies/corr-Ehf.npy")
Evs = np.load("../data/vmcEnergies/corr1-Ev.npy")
Sigvs = np.load("../data/vmcEnergies/corr1-sigma.npy")

print(Sigvs)

N = len(Ehfs)
k = np.linspace(0.1, 2, N, endpoint=True)

fig, ax1 = plt.subplots()
fig.set_size_inches((12, 7))
ax2 = ax1.twinx()

ax1.plot(k, Ehfs, 'ro', label='$E_{HF}$')
ax1.errorbar(k, Evs, Sigvs, marker='.', color='blue', linestyle=' ', label=r'$E_V \pm \sigma_E$')
ax2.plot(k, Ehfs-Evs, 'k--')#, label='Correlation Energy')
# ax2.plot(k, Ehfs-Evs, 'ko--', label='Correlation Energy')
ax2.errorbar(k, Ehfs-Evs, Sigvs, marker='.', color='black', linestyle=' ', label='Correlation Energy')

ax1.legend(loc='lower right')
ax2.legend(loc='lower center')
ax1.set_xlabel("$k$")
ax1.set_ylabel("$E$ [a.u.]")
ax2.set_ylabel(r"$E_c$ [a.u.]")
ax2.set_ylim([0.03, 0.06])

axins1 = ax1.inset_axes([0.07,0.65,0.3,0.3])
# axins2 = axins1.twinx()

axins1.plot(k, Ehfs, 'ro', label='$E_{HF}$')
axins1.errorbar(k, Evs, Sigvs, marker='.', color='blue', linestyle=' ')
# axins2.plot(k, Ehfs-Evs, 'ko--')
# axins1.set_xlabel("$k$")
# axins1.set_ylabel("$E$ [a.u.]")
# axins2.set_ylabel(r"$E_c$ [a.u.]")



ymin,ymax = ax1.get_ylim()

x1, x2, y1, y2 = 1.0, 1.2, ymin + (ymax-ymin)*0.62,ymin + (ymax-ymin)*0.7
axins1.set_xlim([x1, x2])
axins1.set_ylim([y1, y2])
# axins2.set_xlim([x1, x2])

# ymin,ymax = ax2.get_ylim()
# axins2.set_ylim([ymin + (ymax-ymin)*0.4,ymin + (ymax-ymin)*0.7])
ax1.indicate_inset_zoom(axins1)
# ax2.indicate_inset_zoom(axins2)

plt.savefig('../plots/corrPlot.png', bbox_inches='tight')
plt.show()