import os
import sys
import time
from datetime import datetime
import shutil
import math

import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

def ECE(logits, labels, temp_opt=None):
    preds = torch.softmax(logits, 1)
    sorted_pred, sorted_idx = torch.sort(preds, dim=1, descending=True)

    top_i_acc = 0
    top_i_confidence = 0
    ece_list = []

    if temp_opt is not None:
        temps = [float(temp_opt)]
    else:
        temps = [0.5, 1, 2, 4, 8, 16]
    for temp in temps:
        preds = torch.softmax(logits / temp, 1)
        sorted_pred, sorted_idx = torch.sort(preds, dim=1, descending=True)
        i_th_pred = sorted_pred[:,0]
        top_i_confidence = i_th_pred
        i_th_correct = sorted_idx[:,0].eq(labels.data).float()
        top_i_acc = i_th_correct

        n_bins = 19

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)

        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = top_i_confidence.gt(bin_lower.item()) * top_i_confidence.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = top_i_acc[in_bin].float().mean()
                avg_confidence_in_bin = top_i_confidence[in_bin].mean()
                ece += (torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin).item()
        ece_list.append(ece)

    best_ece = min(ece_list)
    arg_best = np.argmin(np.array(ece_list))

    if temp_opt is not None:
        return best_ece
    else:
        return temps[arg_best]

def get_output_file(args):
    start = args.start_index
    end = args.start_index + args.num_samples

    if args.data_type == 'mnli':
        dataset_str = f"{args.data_type}_{args.mnli_option}_{args.attack_target}"
    else:
        dataset_str = args.data_type
    attack_str = args.adv_loss
    if args.adv_loss == 'cw':
        attack_str += f'_kappa={args.kappa}'

    output_file = f"{args.model_name}_{dataset_str}_{start}-{end}"
    output_file += f"_iters={args.num_iters}_{attack_str}_lambda_sim={args.lam_sim}_lambda_perp={args.lam_perp}" \
                   f"_lambda_pref={args.lam_pref}_{args.constraint}.pth"

    return output_file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Logger(object):
    """Reference: https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514"""
    def __init__(self, fn):
        if not os.path.exists("./logs/"):
            os.mkdir("./logs/")

        logdir = 'logs/' + fn
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        if len(os.listdir(logdir)) != 0:
            ans = input("log_dir is not empty. All data inside log_dir will be deleted. "
                            "Will you proceed [y/N]? ")
            if ans in ['y', 'Y']:
                shutil.rmtree(logdir)
            else:
                exit(1)
        self.set_dir(logdir)

    def set_dir(self, logdir, log_fn='log.txt'):
        self.logdir = logdir
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        self.log_file = open(os.path.join(logdir, log_fn), 'a')

    def log(self, string):
        self.log_file.write('[%s] %s' % (datetime.now(), string) + '\n')
        self.log_file.flush()

        print('[%s] %s' % (datetime.now(), string))
        sys.stdout.flush()

    def log_dirname(self, string):
        self.log_file.write('%s (%s)' % (string, self.logdir) + '\n')
        self.log_file.flush()

        print('%s (%s)' % (string, self.logdir))
        sys.stdout.flush()

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

def set_model_path(args, dataset):
    # Naming the saving model
    suffix = "_"
    suffix += str(args.train_type)

    return args.dataset + suffix + 'model'

def save_model(args, model, log_dir, dataset):
    # Save the model
    if isinstance(model, nn.DataParallel):
        model = model.module

    os.makedirs(log_dir, exist_ok=True)
    #model_path = set_model_path(args, dataset)
    save_path = os.path.join(log_dir, 'model')
    torch.save(model.state_dict(), save_path)

def cut_input(tokens):
    attention_mask = (tokens != 1).float()
    max_len = int(torch.max(attention_mask.sum(dim=1)))
    return tokens[:, :max_len]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.average = 0
        self.sum = 0
        self.count = 0

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
