SEED="1234"
GPU=0
DATASET="cola" #cola, emo, hate, spam
cons=1.0 # 1.0, 0.1
div=0.1 # 1.0, 0.1

for dataset in $DATASET
do
for seed in $SEED
do
  #CUDA_VISIBLE_DEVICES=$GPU python train_generative.py --train_type 0315 --base hard --dataset $dataset --seed $seed
  CUDA_VISIBLE_DEVICES=$GPU python train_generative.py --pref_type gen --train_type 0315 --consistency --lambda_cons $cons --lambda_div $div --dataset $dataset --seed $seed
done
done
