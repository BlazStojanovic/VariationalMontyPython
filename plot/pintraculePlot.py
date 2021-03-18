import numpy as np
import matplotlib.pyplot as plt

import matplotlib
from matplotlib import rc
matplotlib.rcParams.update({'font.size': 22})
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{libertine}'

def intracule_r(r):
    return 1.0/(8.0+5.0*np.pi**0.5)*np.square(r)*np.square(1.0+r/2.0)*np.exp(-np.square(r)/4.0)

u = np.load("../data/vmcIntracules/vmc-hf-8g_1e4-pi.npy")

# analytical intracule for k=1/4
r = np.linspace(0, 7, 1000)
uan = intracule_r(r)

# 1e3 samples
fig, ax = plt.subplots(figsize=[7, 7])
u = np.load("../data/vmcIntracules/vmc-hf-8g_1e3-pi.npy")

ax.plot(r, uan, '--', color='red', linewidth=3, label='Analytical')
# ax.hist(u[int(len(u)*0.8):], bins=100, histtype='bar', color='darkorchid', alpha=0.6, density=True, label='VMC, $N={}$'.format(len(u)))
ax.hist(u, bins=50, histtype='bar', color='darkorchid', alpha=0.6, density=True, align='left', label='VMC')

plt.legend()
# ax.hist(u[int(len(u)*0.8):], bins=100, histtype='step', color='black', density=True, label='VMC, $N={}$'.format(len(u)))
ax.hist(u, bins=50, histtype='step', color='black', density=True, align='left', label='VMC, $N={}$'.format(len(u)))
ax.set_xlabel('$u = |r_1-r_2|$')
ax.set_ylabel('$P(u)$')
plt.savefig("../plots/pint1e3-8g1j.png", bbox_inches='tight')


# 1e4 samples
fig, ax = plt.subplots(figsize=[7, 7])
u = np.load("../data/vmcIntracules/vmc-hf-8g_1e4-pi.npy")

ax.plot(r, uan, '--', color='red', linewidth=3, label='Analytical')
# ax.hist(u[int(len(u)*0.8):], bins=100, histtype='bar', color='darkorchid', alpha=0.6, density=True, label='VMC, $N={}$'.format(len(u)))
ax.hist(u, bins=100, histtype='bar', color='darkorchid', alpha=0.6, density=True, align='left', label='VMC')

plt.legend()
# ax.hist(u[int(len(u)*0.8):], bins=100, histtype='step', color='black', density=True, label='VMC, $N={}$'.format(len(u)))
ax.hist(u, bins=100, histtype='step', color='black', density=True, align='left', label='VMC, $N={}$'.format(len(u)))
ax.set_xlabel('$u = |r_1-r_2|$')
ax.set_ylabel('$P(u)$')
plt.savefig("../plots/pint1e4-8g1j.png", bbox_inches='tight')

# 1e5 samples
fig, ax = plt.subplots(figsize=[7, 7])
u = np.load("../data/vmcIntracules/vmc-hf-8g_1e5-pi.npy")

ax.plot(r, uan, '--', color='red', linewidth=3, label='Analytical')
# ax.hist(u[int(len(u)*0.8):], bins=100, histtype='bar', color='darkorchid', alpha=0.6, density=True, label='VMC, $N={}$'.format(len(u)))
ax.hist(u, bins=150, histtype='bar', color='darkorchid', alpha=0.6, density=True, align='left', label='VMC')

plt.legend()
# ax.hist(u[int(len(u)*0.8):], bins=100, histtype='step', color='black', density=True, label='VMC, $N={}$'.format(len(u)))
ax.hist(u, bins=150, histtype='step', color='black', density=True, align='left', label='VMC, $N={}$'.format(len(u)))
ax.set_xlabel('$u = |r_1-r_2|$')
ax.set_ylabel('$P(u)$')
plt.savefig("../plots/pint1e5-8g1j.png", bbox_inches='tight')