"""
From configuration to position intracule

"""

import numpy as np

for i in [3, 4, 5]:
	configs = np.load("../data/vmcConfigurations/vmc-hf-8g-j1_1e{}.npy".format(i))

	# split into r1 and r2 positions
	r1 = configs[:len(configs)//2]
	r2 = configs[len(configs)//2:]

	print(r1)
	print(r2)
	print(np.linalg.norm(r1-r2, axis=1))
	u = np.linalg.norm(r1-r2, axis=1)
	np.save("../data/vmcIntracules/vmc-hf-8g_1e{}-pi.npy".format(i), u)