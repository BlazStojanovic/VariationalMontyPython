import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'


eoptE = np.load('../data/vmcEnergies/vmc-opt-1g-1j-EE1.npy')
eoptV = np.load('../data/vmcEnergies/vmc-opt-1g-1j-Es1.npy')

soptE = np.load('../data/vmcEnergies/vmc-opt-1g-1j-sE1.npy')
soptV = np.load('../data/vmcEnergies/vmc-opt-1g-1j-ss1.npy')

plt.figure(figsize=(12, 7))
plt.xlabel("epochs")
plt.ylabel("$E$ [a.u.]")

plt.plot(eoptE, '-', color='red')
# plt.plot(eoptE+eoptV, '-', color='red', alpha=0.3)
# plt.plot(eoptE-eoptV, '-', color='red', alpha=0.3)
plt.fill_between(np.arange(len(eoptE)), eoptE+eoptV, eoptE-eoptV, alpha=0.2, color='red', label=r'$E$-opt, $E_V \pm \sigma_E$')

plt.plot(soptE, '-', color='blue')
# plt.plot(soptE+soptV, '-', color='blue', alpha=0.3)
# plt.plot(soptE-soptV, '-', color='blue', alpha=0.3)
plt.fill_between(np.arange(len(soptE)), soptE+soptV, soptE-soptV, alpha=0.2, color='blue', label=r'$\sigma$-opt, $E_V \pm \sigma_E$')
plt.ylim([1, 5])
plt.xlim([0, 600])

plt.legend()
plt.savefig('../plots/vmc-opt-1g.png', bbox_inches='tight')
# plt.show()

# plt.semilogy(abs(soptE-2.0))
# plt.show()