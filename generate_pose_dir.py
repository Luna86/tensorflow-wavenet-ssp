import numpy as np
import os
chunk_len = 600
pose = np.loadtxt('/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/training/141126_pose1/s0', delimiter=',')
index = 1
while pose.shape[0] > chunk_len:
	os.mkdir('/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/generate/s{0}'.format(index))
	np.savetxt('/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/generate/s{0}/s{1}'.format(index, index), pose[:chunk_len,:], delimiter=',')
	index += 1
	pose = pose[chunk_len:, :]
