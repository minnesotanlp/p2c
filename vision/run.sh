#!/bin/sh

export DATA=74 # Tested: [74, 88, 89, 91, 93]
export GPU=0
export cons=0.1 # 1.0, 0.1
export div=0.1 #1.0, 0.1

#CUDA_VISIBLE_DEVICES=$GPU python train_vanilla.py --train_type 0315 --data_type $DATA --seed 1
CUDA_VISIBLE_DEVICES=$GPU python train_pref.py --lambda_del $cons --lambda_div $div --pair_loss --train_type 0315 --data_type $DATA --seed 1
 