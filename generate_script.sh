#!/bin/bash
datadir='/luna/data/pose/generate/'
dirlist=`ls $datadir`
for dir in $dirlist
do
	posedir=$datadir$dir
	#python generate_skeleton.py --samples=50 --window=100 --skeleton_out_path=wavenet_generated.mat --motion_seed=$posedir /home/luna/ssp/model/tensorflow-wavenet/logdir/train/2017-01-11T18-20-20/model.ckpt-64999
	python generate_skeleton.py --wavenet_params=wavenet_params1.json --samples=50 --window=100 --skeleton_out_path=/posefs1b/Users/luna/wavenet/generate_skeleton_window100sample100 --motion_seed=$posedir /posefs1b/Users/luna/wavenet/train/2017-01-14T17-14-41/model.ckpt-64999
done






