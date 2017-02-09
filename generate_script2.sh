#!/bin/bash
# pass in two arguments, arg1: timestamp of model, arg2: wavenet_params file used by the model. Fix sample/window size to 100/100 by now. 
modeldir='/media/posefs1b/Users/luna/wavenet/train/'
#datadir='/media/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/generate/'
datadir='/media/posefs1b/Users/hanbyulj/ssp-data/data_xyz_single_rotDec_deltaT_deltaR/haggling3/testing_std_shift100/' 
generatedir='/media/posefs1b/Users/luna/wavenet/generate_skeleton_window200sample100/'
dirlist=`ls $datadir`
#modeldir='/posefs1b/Users/luna/wavenet/train/'
timestamp=$1
params=$2
ckpt='/model.ckpt-64999'
for dir in $dirlist
do
	posedir=$datadir$dir
        CUDA_VISIBLE_DEVICES=1 python generate_skeleton.py --wavenet_params=$params --samples=100 --window=200 --skeleton_out_path=$generatedir --motion_seed=$posedir $modeldir$timestamp$ckpt
done



