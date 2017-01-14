import numpy as np
pose = np.loadtxt('/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/training/141126_pose1/s0', delimiter=',')
index = 1
while pose.shape[0] > 100:
	np.savetxt('/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/generate_chunk/s{0}'.format(index), pose[:100,:], delimiter=',')
	index += 1
	pose = pose[100:, :]
