"""
Project: VQMC - Variational Quantum Monte Carlo for a 3d "Hookium"
Author: Blaz Stojanovic
Date: 11/02/2021
----------
Contains: Analytical expressions for Hookium (k=1/4, E=2), that will be used to compare results to simulations
"""

import numpy as np


def wfunction_r(r1, r2):
    """
    Normalized wave-function in position space.
    
    Parameters
    ----------
    r1, r2: np.arrays
        positions to be evaluated at

    Returns
    ----------
    $\Psi(r_1, r_2)$, wavefunction evaluated at positions r1, r2

    """

    # calculate pointwise distances
    r12 = np.linalg.norm(r1-r2) # TODO doublecheck
    
    # calculate norms of points
    R1 = np.linalg.norm(r2, axis=1)
    # R1 = np.linalg.norm(r1, axis=1) # TODO doublecheck
    R2 = np.linalg.norm(r2, axis=1) # TODO doublecheck

    return 1.0/(2.0*np.sqrt(8.0*np.power(np.pi, 5.0/2.0) + 5.0*np.power(np.pi, 3.0)))*(1+r12)*np.exp(-1.0/4.0*(np.square(R1) + np.square(R2)))

def intracule_r(u):
    return 1.0/(8+5*np.pi**0.5)*np.square(u)*(1+np.square(u)/2.0)*np.exp(-np.square(u)/4)


def intracule_wigner_approximate(u, v):
    pass


def wfunction_p(p1, p2):
    pass
    

def intracule_p(v):
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # u = np.linspace(0, 10, 100)
    # plt.plot(u, intracule_r(u), '.-')
    # plt.show()

    N = 20
    dx = np.linspace(-1, 1, N)
    dy = np.linspace(-1, 1, N)
    dz = np.linspace(0, 1, N)

    pc = np.zeros((N, N, N, 3))

    for i, x in enumerate(dx):
        for j, y in enumerate(dy):
            for k, z in enumerate(dz):
                # print(i, j, k)
                pc[i, j, k, :] = [x, y, z]

    pc = pc.reshape((N**3, 3))
    print(np.shape(pc))
    psi = wfunction_r(np.array([20.0, 0, 0]).T, pc)

    # plt.plot(pc[:, 2])

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # p = ax.scatter(pc[:, 0], pc[:, 1], zs=pc[:, 2], c=psi)
    # fig.colorbar(p)
    # plt.show()    

    import plotly.graph_objects as go
    fig=go.Figure(data=go.Volume(
        x=pc[:, 0],
        y=pc[:, 1],
        z=pc[:, 2],
        value=psi,
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=17, # needs to be a large number for good volume rendering
    ))

    fig.show()
