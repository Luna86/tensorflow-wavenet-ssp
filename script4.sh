CUDA_VISIBLE_DEVICES=3 python train_skeleton.py --batch_size=50 --data_dir=/home/lunay/data/pose/training \
<<<<<<< HEAD
--checkpoint_every=1000 --num_steps=65000 --sample_size=500 --learning_rate=1e-4 --optimizer=adam \
--histograms=False --logdir_root=/media/posefs1b/Users/luna/wavenet --wavenet_params=./wavenet_params4.json
=======
--checkpoint_every=1000 --num_steps=65000 --sample_size=500 --learning_rate=1e-3 --optimizer=adam \
--histograms=False --wavenet_params=./wavenet_params1.json --epsilon=1e-4
>>>>>>> 650c63112c4e7865ce130d00690baf98b973ac43
