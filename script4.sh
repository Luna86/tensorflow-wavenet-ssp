CUDA_VISIBLE_DEVICES=3 python train_skeleton.py --batch_size=50 --data_dir=/home/lunay/data/pose/training \
--checkpoint_every=1000 --num_steps=65000 --sample_size=500 --learning_rate=1e-4 --optimizer=adam \
--histograms=False --logdir_root=/media/posefs1b/Users/luna/wavenet --wavenet_params=./wavenet_params4.json
