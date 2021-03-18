
"""
Using answer from:
https://stackoverflow.com/questions/25286811/how-to-plot-a-3d-density-map-in-python-with-matplotlib
"""

import matplotlib.pyplot as plt
from scipy import stats
from fastkde import fastKDE
import multiprocessing

from mayavi import mlab
import numpy as np

for i in [5]:
	configurations = np.load("../data/vmcConfigurations/vmc-hf-8g-j1_1e{}.npy".format(i))

	x, y, z = configurations[:, 0], configurations[:, 1], configurations[:, 2]

	kde = stats.gaussian_kde(configurations.T)
	density = kde(configurations.T)

	# Plot scatter with mayavi
	figure = mlab.figure('DensityPlot', bgcolor=(1,1,1), fgcolor=(0,0,0), size=(1000, 1000))
	pts = mlab.points3d(configurations[:, 0], configurations[:, 1], configurations[:, 2], density, scale_mode='none', scale_factor=0.05, opacity=0.4)

	# # Evaluate kde on a grid
	xmin, ymin, zmin = x.min(), y.min(), z.min()
	xmax, ymax, zmax = x.max(), y.max(), z.max()
	xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
	coords = np.vstack([item.ravel() for item in [xi, yi, zi]]) 
	density = kde(coords).reshape(xi.shape)

	# Plot scatter with mayavi
	grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
	min = density.min()
	max=density.max()
	mlab.pipeline.volume(grid, vmin=min, vmax=min + .5*(max-min))

	# mlab.axes(extent=[-4, 4,-4, 4,-4, 4])
	mlab.orientation_axes()
	mlab.view(distance=25)
	mlab.savefig("../plots/wf-8g-1j-1e{}.png".format(i))
	mlab.close()
