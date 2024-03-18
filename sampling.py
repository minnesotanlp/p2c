import os
import sys
import time
from datetime import datetime
import shutil
import math
import pickle

import numpy as np
import torch
import random
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical

def select_ensemble(args, all_pref_preds, all_probs_soft, labels, epoch):
    if 'dynasent' in args.dataset:
        label_str = ['negative', 'positive', 'neutral']
    elif 'polite' in args.dataset:
        label_str = ['impolite', 'polite']
    elif 'mnli' in args.dataset:
        label_str = ['entailment', 'neutral', 'contradiction']
    else:
        label_str = ['nonoffensive', 'offensive']

    with open('./{}/{}_human_pref.pkl'.format(args.pre_gen, args.dataset), 'rb') as f:
        human_preference_all = pickle.load(f)

    with open('./{}/{}_indices.pkl'.format(args.pre_gen, args.dataset), 'rb') as f:
        indices_all = pickle.load(f)

    selected_idx = torch.zeros(len(all_pref_preds[0]), 2).long()
    soft_labels = torch.Tensor(np.load('./{}/{}_soft_label.npy'.format(args.pre_gen, args.dataset)))

    all_probs = all_probs_soft[torch.arange(len(soft_labels)), labels]

    for label in label_str:
        indices_label = indices_all[label]
        prob = all_probs[indices_label]

        if args.sampling == 'inconsistency':
            prob_delta = (all_probs_soft[indices_label].unsqueeze(1) - all_probs_soft[indices_label].unsqueeze(0))  # N x N x K
            soft_labels_delta = (soft_labels[indices_label].unsqueeze(1) - soft_labels[indices_label].unsqueeze(0))

            # Due to memory issue, considering element-wise operation instead of matrix-level
            for i in range(len(prob_delta)):
                zeros = torch.zeros(len(prob_delta)).float()
                mask1, mask2 = (soft_labels_delta[i] >= 0).float(), (soft_labels_delta[i] < 0).float()
                loss_delta = (mask1 * torch.max(zeros.unsqueeze(1), soft_labels_delta[i] - prob_delta[i])).sum(dim=-1)
                loss_delta += (mask2 * torch.max(zeros.unsqueeze(1), prob_delta[i] - soft_labels_delta[i])).sum(dim=-1)

                loss_delta_norm = (loss_delta - loss_delta.min()) / (loss_delta.max() - loss_delta.min())

                dist = Categorical(loss_delta_norm)
                select_idx = dist.sample()
                selected_idx[indices_label[i], 0] = indices_label[select_idx]
                selected_idx[indices_label[i], 1] = human_preference_all[label][i, select_idx]
        else:
            candidate_idx = indices_all[label]
            human_preference_label = human_preference_all[label]

            converts = []
            mats = []
            for i in range(len(all_pref_preds)):
                pref = all_pref_preds[i][candidate_idx]
                mat = pref.unsqueeze(1) - pref.unsqueeze(0)
                convert = 2 * (mat == 0) + 1 * (mat > 0) + torch.eye(len(candidate_idx))
                converts.append(convert)
                mats.append(mat.unsqueeze(0))

            disagree = 0
            for i in range(len(converts)):
                for j in range(i + 1, len(converts)):
                    disagree += (converts[i] != converts[j])

            wrong = 0
            for i in range(len(converts)):
                wrong += (converts[i] != human_preference_label)

            score = disagree
            score_max = score.max(dim=1)[0]

            for i in range(len(candidate_idx)):
                score_i = (score[i] == score_max[i]).nonzero()[:, 0]
                selected = score_i[torch.randint(0, len(score_i), (1,))]

                if args.anneal:
                    p_anneal = epoch / args.epochs

                    if torch.bernoulli(torch.Tensor([p_anneal])):
                        selected = selected
                    else:
                        selected = torch.randint(0, len(candidate_idx), (1,))
                        while selected == i:
                            selected = torch.randint(0, len(candidate_idx), (1,))

                selected_idx[candidate_idx[i], 0] = candidate_idx[selected]
                selected_idx[candidate_idx[i], 1] = human_preference_label[i, selected]

    print("Number of preference 0: {}, 1: {}, 2:{}".format((selected_idx[:, 1] == 0).float().sum(),
                                                           (selected_idx[:, 1] == 1).float().sum(),
                                                           (selected_idx[:, 1] == 2).float().sum()))
    return selected_idx[:, 0], selected_idx[:, 1]