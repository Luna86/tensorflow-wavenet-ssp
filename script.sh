python train_skeleton.py --batch_size=50 --data_dir=/posefs1b/Users/hanbyulj/ssp-data/data_lunaForm/pose/training \
--checkpoint_every=1000 --num_steps=65000 --sample_size=200 --learning_rate=1e-3 --optimizer=adam \
--histograms=False --wavenet_params=./wavenet_params.json
