"""
Paper plot: Hartree-Fock energy, comparison between fully variational approach with different numbers of gaussians
and literature results from Neill and Gill 2003.
"""

import matplotlib.pyplot as plt
import numpy as np


# setup pyplot
import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'


# Variational HF data
e2g = np.load("../data/hfEnergy/hf2g.npy")
e3g = np.load("../data/hfEnergy/hf3g.npy")
e4g = np.load("../data/hfEnergy/hf4g.npy")
e5g = np.load("../data/hfEnergy/hf5g.npy")
e6g = np.load("../data/hfEnergy/hf6g.npy")
e8g = np.load("../data/hfEnergy/hf8g.npy")
# e10g = np.load("../data/hfEnergy/hf10g.npy")

# Literature data
ho1 = 2.06418958
ho3 = 2.03845337
ho5 = 2.03843889
ho7 = 2.03843887

fig, ax = plt.subplots(figsize=[12, 7])

# plt.yscale('log')
# plot analytical results
ax.plot([0, len(e6g[0])], [ho1, ho1], 'k*--', label='$n_{HO} = 1$')
ax.plot([0, len(e6g[0])], [ho3, ho3], 'ks--', label='$n_{HO} = 3$')
ax.plot([0, len(e6g[0])], [ho5, ho5], 'k<--', label='$n_{HO} = 5$')
ax.plot([0, len(e6g[0])], [ho7, ho7], 'ko--', label='$n_{HO} = 7$')

ax.plot(e2g[0], 'o--', label='sto-2g')
ax.plot(e3g[0], 'o--', label='sto-3g')
ax.plot(e4g[0], 'o--', label='sto-4g')
ax.plot(e5g[0], 'o--', label='sto-5g')
ax.plot(e6g[0], 'o--', label='sto-6g')
ax.plot(e8g, 'o--', label='sto-8g')
# ax.plot(e10g[0], 'o--', label='sto-10g')

axins = ax.inset_axes([0.25,0.35,0.5,0.55])
x1, x2, y1, y2 = 10, 22, 2.038433, 2.03847

# axins.plot([0, len(e6g[0])], [ho1, ho1], 'k*--', label='$n_{HO} = 1$')
axins.plot([0, len(e6g[0])], [ho3, ho3], 'ks--', label='$n_{HO} = 3$')
axins.plot([0, len(e6g[0])], [ho5, ho5], 'k<--', label='$n_{HO} = 5$')
axins.plot([0, len(e6g[0])], [ho7, ho7], 'ko--', label='$n_{HO} = 7$')

axins.plot(e2g[0], 'o--', label='sto-2g')
axins.plot(e3g[0], 'o--', label='sto-3g')
axins.plot(e4g[0], 'o--', label='sto-4g')
axins.plot(e5g[0], 'o--', label='sto-5g')
axins.plot(e6g[0], 'o--', label='sto-6g')
axins.plot(e8g, 'o--', label='sto-8g')
# axins.plot(e10g[0], 'o--', label='sto-10g')

# axins.set_yscale('log')
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.legend()

ax.indicate_inset_zoom(axins)

ax.set_xlabel("epoch")
ax.set_ylabel("$E_{HF}$")

plt.ylim([2.038, 2.045])
plt.savefig("../plots/HF_optimization.png", bbox_inches='tight')
plt.show()
# plt.close()