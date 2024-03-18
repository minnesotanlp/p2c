SEED="111"
GPU=0
DATASET="offensive" #dynasent1 dynasent2 polite_wiki2 polite_stack2 offensive mnli
cons=1.0 # 1.0, 0.1
div=0.1 # 1.0, 0.1

for dataset in $DATASET
do
for seed in $SEED
do
  #CUDA_VISIBLE_DEVICES=$GPU python train_extractive.py --train_type 0316 --base hard --pre_gen final_files --dataset $dataset --seed $seed

  # With hard pair sampling
  CUDA_VISIBLE_DEVICES=$GPU python train_extractive.py --train_type 0316 --pre_gen final_files --sampling disagreement --pair_loss --lambda_cons $cons --lambda_div $div --dataset $dataset --seed $seed

  # Without hard pair sampling
  #CUDA_VISIBLE_DEVICES=$GPU python train_extractive.py --train_type 0316 --pre_gen final_files --pair_loss --lambda_cons $cons --lambda_div $div --dataset $dataset --seed $seed
done
done
