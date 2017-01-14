import numpy as np
import os
pose = np.loadtxt('/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/training/141126_pose1/s0', delimiter=',')
index = 1
while pose.shape[0] > 500:
	os.mkdir('/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/generate/s{0}'.format(index))
	np.savetxt('/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/generate/s{0}/s{1}'.format(index, index), pose[:500,:], delimiter=',')
	index += 1
	pose = pose[500:, :]
