# Prefer to Classify: Improving Text Classifiers via Auxiliary Preference Learning
This repository provides datasets and codes of the following paper:

> [Prefer to Classify: Improving Text Classifiers via Auxiliary Preference Learning](https://arxiv.org/abs/2306.04925) <br>
> [Jaehyung Kim](https://sites.google.com/view/jaehyungkim), Jinwoo Shin, [Dongyeop Kang](https://dykang.github.io/) <br>
> [ICML 2023](https://icml.cc/) <br>

<p align="center" >
    <img src=assets/icml23_main_figure.jpg width="100%">
</p>

## Installation
The following command installs all necessary packages:
```
pip install -r requirements.txt
```
The project was tested using `Python 3.8`.

Also, one needs to download the pre-generated files from [Google Drive](https://drive.google.com/file/d/1HmdFMtSwHe0FSSi9xOCkCC_IHsDWLV7Z/view?usp=sharing), and locate them in the folder at `args.pre_gen`.

## 0. Overview of Prefer to Classify (P2C)

P2C is a multi-task learning framework that jointly learns (1) a target task (e.g., sentiment classification) and (2) a preference between two samples for the given task label (e.g., positive or negative). From the pair-wise comparison, P2C captures the finer task information which can't be captured in a sample-wise evaluation. 

To train the classifier with P2C, one first needs to prepare a target dataset consisting of (inputs and labels). Next, one needs to collect preference labels using one of the proposed three different ways (described in below). Finally, one can train the classifier using both target task labels and collected preference labels. 

To ease the usage of P2C, we release the constructed datasets (including preference labels) and checkpoints of the classifiers (trained by P2C) at [Huggingface](https://huggingface.co/JaehyungKim) or [Google Drive](https://drive.google.com/drive/folders/1za5ZeyMFxt86ad8xWcCnsCtZt-JA2hxQ?usp=sharing). More details are described in the below parts. 

## 1. P2C with Generative Preference

The first proposed way to collect preference labels is prompting the recent large language models (LLMs) to ask which sentence is more preferred as a specific task label. One can collect the generative preference labels using the following scripts:
```
python query_gpt.py --api sk-xxxxx
```
In the case of the used datasets in the paper, they are uploaded in Huggingface (including preference labels) and automatically downloaded when running the code to train the classifier.

Consequently, one can train the classifier with P2C using generative preference as follow:
```
python train_generative.py --pref_type gen --train_type xxxx --consistency --lambda_cons $cons --lambda_div $div --dataset $dataset --seed $seed
```
For more details and running the baseline, please see the script `run_generative.sh`.

## 2. P2C with Subjective Preference

The second proposed way to collect preference labels is by hiring crowd workers. As denoted in the paper, we use Amazon Mechanical Turk (AMT) and collect the subjective preference labels on 5000 pair of samples on Dynasent2 Benchmark. To ease the experiments, we release the Huggingface dataset including other preference labels (extractive and generative). The dataset and the desired preference labels are automatically downloaded when running the following code to train the classifier with P2C: 
```
python train_subjective.py --train_type xxxx --pref_type sub --consistency --lambda_cons $cons --lambda_div $div --dataset $dataset --seed $seed
```
For more details and running the baseline, please see the script `run_subjective.sh`.

## 3. P2C with Extractive Preference

The third proposed way to collect preference labels is recycling the annotation records for the target task. For example, when 5 annotators are engaged to determine the label for the binary sentiment classification, it is natural to assume that the sample with more votes as positive label is preferred as positive as well. The details how to extract this preference label from the existing annotation records are described in `pref_gen.ipynb`.

We remark that the used datasets in the paper are uploaded in Huggingface (including preference labels) and automatically downloaded when running the code to train the classifier.

Consequently, one can train the classifier with P2C using extractive preference as follows:
```
python train_extractive.py --train_type xxxx --pre_gen final_files --sampling disagreement --pair_loss --lambda_cons $cons --lambda_div $div --dataset $dataset --seed $seed
```
For more details and running the baseline, please see the script `run_extractive.sh`.

## 4. P2C on Vision

In addition, we have demonstrated the effectiveness of P2C on image classification. Specifically, we construct the multiple binary classification benchmarks using SUN dataset. First, one needs to pre-process [SUN dataset](https://drive.google.com/file/d/1WW3fFjbYwFeZ4x_lo55Ip_JzlzURT4YU/view?usp=sharing) using `vision/SUN_preprocessing.ipynb`. Then, one can run the following script to train ResNet-18 with P2C using extractive preference labels:
```
python train_pref.py --lambda_del $cons --lambda_div $div --pair_loss --train_type xxxx --data_type $DATA --seed 1
```
For more details and running the baseline, please see the script `vision/run.sh`. Also, the trained checkpoints could be downloaded in [Google Drive](https://drive.google.com/drive/folders/11weoj43nODwVoHT5PXsmzQsGFmZCyiYf?usp=sharing). 


## Citation
If you find this work useful for your research, please cite our papers:

```
@article{kim2023p2c,
  title={Prefer to Classify: Improving Text Classifiers via Auxiliary Preference Learning},
  author={Kim, Jaehyung and Shin, Jinwoo and Kang, Dongyeop},
  journal={Proceedings of the 40th International Conference on Machine Learning (ICML)},
  year={2023}
}
```
