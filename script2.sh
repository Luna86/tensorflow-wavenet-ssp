CUDA_VISIBLE_DEVICES=1 python train_skeleton.py --batch_size=50 --data_dir=/home/lunay/data/pose/training \
--checkpoint_every=1000 --num_steps=65000 --sample_size=500 --learning_rate=1e-3 --optimizer=adam \
--histograms=False --wavenet_params=./wavenet_params1.json --epsilon=0.5
