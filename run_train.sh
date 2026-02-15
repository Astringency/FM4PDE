#!/bin/bash

torchrun --master_addr=127.0.0.1 --master_port=12356 --nproc_per_node=4 train.py --batch_size=10 --epochs=500 --accum_iter=256 --decay_lr --dataset="darcy"
torchrun --master_addr=127.0.0.1 --master_port=12355 --nproc_per_node=4 train.py --batch_size=10 --epochs=500 --accum_iter=256 --decay_lr --dataset="poisson"
torchrun --master_addr=127.0.0.1 --master_port=12354 --nproc_per_node=2 train.py --batch_size=10 --epochs=500 --accum_iter=256 --decay_lr --dataset="helmholtz"
torchrun --master_addr=127.0.0.1 --master_port=12353 --nproc_per_node=4 train.py --batch_size=10 --epochs=500 --accum_iter=256 --decay_lr --dataset="nsnonbounded"
torchrun --master_addr=127.0.0.1 --master_port=12352 --nproc_per_node=4 train.py --batch_size=10 --epochs=500 --accum_iter=256 --decay_lr --dataset="shallow_water"
torchrun --master_addr=127.0.0.1 --master_port=12351 --nproc_per_node=2 train.py --batch_size=2 --epochs=500 --accum_iter=512 --decay_lr --dataset="reaction_diffusion"

