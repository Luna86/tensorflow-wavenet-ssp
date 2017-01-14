#!/bin/bash
datadir='/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/generate/'
dirlist=`ls $datadir`
for dir in $dirlist
do
	posedir=$datadir$dir
	python generate_skeleton.py --samples=50 --window=100 --skeleton_out_path=wavenet_generated.mat --motion_seed=$posedir /home/luna/ssp/model/tensorflow-wavenet/logdir/train/2017-01-11T18-20-20/model.ckpt-64999
done






