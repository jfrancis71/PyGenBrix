python generative_trainer.py --rv_distribution=bernoulli --model=PixelCNN --dataset=mnist32 --train_log_freq=0 --trainer=ema --ema_rate=0.99 --tensorboard_log=./mnist32/pixelcnn/ema_99
python generative_trainer.py --rv_distribution=bernoulli --model=VDVAE --dataset=mnist32 --train_log_freq=0 --trainer=ema --ema_rate=0.99 --tensorboard_log=./mnist32/vdvae/ema_99
