#!/bin/bash
# pass in two arguments, arg1: timestamp of model, arg2: wavenet_params file used by the model. Fix sample/window size to 100/100 by now. 
modeldir='/media/posefs1b/Users/luna/wavenet/train/'
datadir='/media/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/generate_shift1/'
generatedir='/media/posefs1b/Users/luna/wavenet/generate_shift1_skeleton_window100sample100/'
dirlist=`ls $datadir`
#modeldir='/posefs1b/Users/luna/wavenet/train/'
timestamp=$1
params=$2
ckpt='/model.ckpt-64999'
for dir in $dirlist
do
	posedir=$datadir$dir
	#python generate_skeleton.py --wavenet_params=wavenet_params1.json --samples=100 --window=500 --skeleton_out_path=wavenet_generated.mat --motion_seed=$posedir /home/luna/ssp/model/tensorflow-wavenet-ssp/logdir-server6/train/2017-01-15T07-10-00/model.ckpt-64999.data-00000-of-00001
        CUDA_VISIBLE_DEVICES=2 python generate_skeleton.py --wavenet_params=$params --samples=100 --window=100 --skeleton_out_path=$generatedir --motion_seed=$posedir $modeldir$timestamp$ckpt
done



