DATASET="dynasent2_sub_subjective" # dynasent2_sub_subjective, dynasent2_sub_extractive, dynasent2_sub_generative
SEED="111"
GPU=7
cons=1.0 # 1.0, 0.1
div=0.1 # 1.0, 0.1

for dataset in $DATASET
do
for seed in $SEED
do
  #CUDA_VISIBLE_DEVICES=$GPU python train_subjective.py --train_type 0316 --base hard --dataset $dataset --seed $seed
  CUDA_VISIBLE_DEVICES=$GPU python train_subjective.py --train_type 0316 --pref_type sub --consistency --lambda_cons $cons --lambda_div $div --dataset $dataset --seed $seed
done
done
