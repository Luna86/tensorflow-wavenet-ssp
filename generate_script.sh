#!/bin/bash
datadir='/luna/data/pose/generate/'
dirlist=`ls $datadir`
for dir in $dirlist
do
	posedir=$datadir$dir
	python generate_skeleton.py --samples=50 --window=100 --skeleton_out_path=wavenet_generated.mat --motion_seed=$posedir /home/lunay/tensorflow-wavenet_ssp/logdir/train/2017-01-14T17-18-27
done






