import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import random
import datetime
import pickle
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from sklearn.metrics import matthews_corrcoef, f1_score
from tqdm import tqdm

from utils import Logger, set_seed, set_model_path, save_model, AverageMeter, cut_input, ECE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = './checkpoint'

def diversity_loss(out_pref1, out_pref2):
    pref_probs_all = []
    n_ensemble = len(out_pref1)
    for i in range(n_ensemble):
        pref_probs_i = torch.cat([torch.exp(out_pref2[i]), torch.exp(out_pref1[i])], dim=-1)  # pref: 1 if x1 > x2, 0 else
        pref_probs_i = pref_probs_i / (torch.exp(out_pref2[i]) + torch.exp(out_pref1[i])).sum(dim=-1, keepdim=True)

        pref_probs_all.append(pref_probs_i)

    pref_sim = 0
    for i in range(n_ensemble):
        for j in range(n_ensemble):
            if i != j:
                pref_sim += (-1 * pref_probs_all[i].data * torch.log(pref_probs_all[j] + 1e-8)).sum(dim=-1).mean()

    loss_div = pref_sim / (n_ensemble * (n_ensemble - 1))
    return loss_div

def train_base(args, loader, model, optimizer, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()

    criterion = nn.CrossEntropyLoss(reduction='none')

    for batch_idx, (tokens, tokens2, labels, _, indices) in enumerate(tqdm(loader)):
        batch_size = tokens.size(0)

        tokens = cut_input(tokens)
        tokens = tokens.to(device)
        labels = labels.to(device)

        out_cls = model(tokens)
        if args.base == 'cskd':
            loss = criterion(out_cls, labels).mean()

            tokens2 = cut_input(tokens2)
            tokens2 = tokens2.to(device)

            with torch.no_grad():
                teacher_cls, _ = model(tokens2)
            teacher_probs = (teacher_cls / args.temperature).softmax(dim=-1)
            probs = (out_cls / args.temperature).softmax(dim=-1)

            loss += (args.temperature ** 2) * (-1 * teacher_probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        elif args.base == 'ls':
            soft_labels = ((1 - args.tau) / (args.n_class - 1)) * torch.ones(batch_size, int(args.n_class)).cuda()
            soft_labels[torch.arange(batch_size), labels] = args.tau

            probs = out_cls.softmax(dim=-1)
            loss = (-1 * soft_labels * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        elif args.base == 'max_ent':
            loss = criterion(out_cls, labels).mean()

            probs = out_cls.softmax(dim=-1)
            ent = (-1 * probs.detach() * torch.log(probs + 1e-8)).sum(dim=-1).mean()

            loss += (-1 * args.lambda_ent * ent)
        else:
            loss = criterion(out_cls, labels).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # cls_acc
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        losses['cls'].update(loss.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f]' % (epoch, losses['cls_acc'].average, losses['cls'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

def train_preference(args, loader, model, optimizer, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()
    losses['pref'] = AverageMeter()
    losses['pref_acc'] = AverageMeter()
    losses['div'] = AverageMeter()
    losses['delta'] = AverageMeter()
    losses['consistency'] = AverageMeter()

    criterion = nn.CrossEntropyLoss(reduction='none')
    
    n_samples = args.n_samples
    # Logs 
    prefs_all = torch.zeros(3, n_samples)
    probs_all = torch.zeros(n_samples, args.n_class)
    pref_iter = iter(loader)

    for batch_idx, (tokens1, tokens2, labels, pref, indices) in enumerate(tqdm(loader)):
        batch_size = tokens1.size(0)
        tokens1, tokens2, labels, pref = tokens1, tokens2, labels.to(device), pref.to(device)
        
        # Processing of inputs and task labels
        tokens1 = cut_input(tokens1).to(device)
        tokens2 = cut_input(tokens2).to(device)

        # Processing of preference labels
        pref1, pref2 = pref.clone(), pref.clone()
        pref1[pref1 == 2] = 0
        pref2[pref2 == 2] = 1
        pref_label = torch.zeros(batch_size, 2).cuda()
        pref_label[torch.arange(batch_size), pref1] += 0.5
        pref_label[torch.arange(batch_size), pref2] += 0.5

        out_cls, out_pref1 = model(tokens1, y=labels, pref=True)
        out_cls2, out_pref2 = model(tokens2, y=labels, pref=True)

        # Classification loss
        loss_cls = (0.5 * criterion(out_cls, labels) + 0.5 * criterion(out_cls2, labels)).mean()
        
        probs1, probs2 = out_cls.softmax(dim=-1), out_cls2.softmax(dim=-1)
        zeros = torch.zeros(batch_size).float().to(device)

        # Consistency losses between cls heads and pref heads
        if args.consistency:
            mask1, mask2, mask3 = (pref == 1).float(), (pref == 0).float(), (pref == 2).float()
            prob_delta = (probs1 - probs2)[torch.arange(batch_size), labels] # If all is used, then it should be changed

            loss_cons = (mask1 * torch.max(zeros, -1 * prob_delta))
            loss_cons += (mask2 * torch.max(zeros, prob_delta))
            loss_cons += (mask3 * torch.max(zeros, prob_delta.abs()))
            loss_cons = loss_cons.mean()
        else:
            loss_cons = torch.Tensor([0]).cuda().mean()

        # Preference losses for multiple heads
        loss_pref, pref_probs = 0, 0
        pref_probs_all = []
        for i in range(len(out_pref1)):
            pref_probs_i = torch.cat([torch.exp(out_pref2[i]), torch.exp(out_pref1[i])], dim=-1)  # pref: 1 if x1 > x2, 0 else
            pref_probs_i = pref_probs_i / (torch.exp(out_pref2[i]) + torch.exp(out_pref1[i])).sum(dim=-1, keepdim=True)

            loss_pref += (-1 * pref_label * torch.log(pref_probs_i + 1e-8)).sum(dim=-1).mean()
            pref_probs += pref_probs_i
            pref_probs_all.append(pref_probs_i)
        loss_pref /= len(out_pref1)
        pref_probs /= len(out_pref1)

        # Diversity between preference heads
        loss_div = diversity_loss(out_pref1, out_pref2)
        loss = loss_cls + args.lambda_cons * loss_cons + args.lambda_pref * loss_pref - args.lambda_div * loss_div
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # cls_acc
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum() / batch_size

        # pref_acc
        _, pred_pref = pref_probs.max(dim=1)
        n_pref = (pref != 2).float().sum()
        corrects_pref = (pred_pref == pref)[pref != 2].float()
        acc_pref = corrects_pref.sum() / (1e-8 + n_pref)

        losses['cls'].update(loss_cls.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)
        losses['pref'].update(loss_pref.item(), batch_size)
        losses['pref_acc'].update(acc_pref.item(), batch_size)
        losses['delta'].update(loss_div.item(), batch_size)

        p_delta = ((probs1 - probs2)[torch.arange(batch_size), labels] >= 0).data
        consistency = (pref == p_delta)[pref != 2].float().mean()
        losses['consistency'].update(consistency.item(), n_pref)

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f] [AccP %.3f] [LossP %.3f] [LossDel %.3f] [Consist %.3f]' \
          % (epoch, losses['cls_acc'].average, losses['cls'].average, losses['pref_acc'].average,
             losses['pref'].average, losses['delta'].average, losses['consistency'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

    return prefs_all, probs_all

def train_preference_extractive(args, loader, pair_idx, model, optimizer, epoch=0, logger=None):
    model.train()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['cls_acc'] = AverageMeter()
    losses['pref'] = AverageMeter()
    losses['pref_acc'] = AverageMeter()
    losses['div'] = AverageMeter()
    losses['delta'] = AverageMeter()
    losses['consistency'] = AverageMeter()

    criterion = nn.CrossEntropyLoss(reduction='none')
    soft_labels = torch.Tensor(np.load('./{}/{}_soft_label.npy'.format(args.pre_gen, args.dataset))).to(device)

    prefs_all = torch.zeros(3, args.n_samples)
    probs_all = torch.zeros(args.n_samples, args.n_class)
    saved_probs = torch.zeros(3, args.n_samples, 2)

    for batch_idx, (tokens1, tokens2, labels, pref, indices) in enumerate(tqdm(loader)):
        batch_size = tokens1.size(0)
        tokens1, tokens2, labels, pref = tokens1.to(device), tokens2.to(device), labels[:,0].to(device), pref.to(device)
        tokens1, tokens2 = cut_input(tokens1), cut_input(tokens2)

        pref1, pref2 = pref.clone(), pref.clone()
        pref1[pref1 == 2] = 0
        pref2[pref2 == 2] = 1
        pref_label = torch.zeros(batch_size, 2).cuda()
        pref_label[torch.arange(batch_size), pref1] += 0.5
        pref_label[torch.arange(batch_size), pref2] += 0.5

        out_cls, out_pref1 = model(tokens1, y=labels, pref=True)
        out_cls2, out_pref2 = model(tokens2, y=labels, pref=True)
    
        loss_cls = 0.5 * criterion(out_cls, labels).mean() + 0.5 * criterion(out_cls2, labels).mean()
        probs1, probs2 = out_cls.softmax(dim=-1), out_cls2.softmax(dim=-1)
        zeros = torch.zeros(batch_size).float().to(device)

        if args.pair_loss:
            soft_labels_batch, soft_labels_batch2 = soft_labels[indices], soft_labels[pair_idx[indices]]

            # Delta
            soft_labels_delta, prob_delta = (soft_labels_batch - soft_labels_batch2), (probs1 - probs2)
            mask1, mask2 = (soft_labels_delta >= 0).float(), (soft_labels_delta < 0).float()
            loss_delta = (mask1 * torch.max(zeros.unsqueeze(1), soft_labels_delta - prob_delta)).sum(dim=-1)
            loss_delta += (mask2 * torch.max(zeros.unsqueeze(1), prob_delta - soft_labels_delta)).sum(dim=-1)
            loss_delta = loss_delta.mean()
        elif args.consistency:
            mask1, mask2, mask3 = (pref == 1).float(), (pref == 0).float(), (pref == 2).float()
            prob_delta = (probs1 - probs2)[torch.arange(batch_size), labels] # If all is used, then it should be changed

            loss_delta = (mask1 * torch.max(zeros, -1 * prob_delta))
            loss_delta += (mask2 * torch.max(zeros, prob_delta))
            loss_delta += (mask3 * torch.max(zeros, prob_delta.abs()))
            loss_delta = loss_delta.mean()
        else:
            loss_delta = torch.Tensor([0]).cuda().mean()

        loss_pref, pref_probs = 0, 0
        pref_probs_all = []
        for i in range(len(out_pref1)):
            pref_probs_i = torch.cat([torch.exp(out_pref2[i]), torch.exp(out_pref1[i])], dim=-1)  # pref: 1 if x1 > x2, 0 else
            pref_probs_i = pref_probs_i / (torch.exp(out_pref2[i]) + torch.exp(out_pref1[i])).sum(dim=-1, keepdim=True)

            loss_pref += (-1 * pref_label * torch.log(pref_probs_i + 1e-8)).sum(dim=-1).mean()
            pref_probs += pref_probs_i
            pref_probs_all.append(pref_probs_i)
        loss_pref /= len(out_pref1)
        pref_probs /= len(out_pref1)

        loss_div = diversity_loss(out_pref1, out_pref2)

        saved_probs[0, indices] = pref_probs_all[0].data.cpu()
        saved_probs[1, indices] = pref_probs_all[1].data.cpu()
        saved_probs[2, indices] = pref_probs_all[2].data.cpu()

        loss = args.lambda_cls * loss_cls + args.lambda_del * loss_delta + args.lambda_pref * loss_pref - args.lambda_div * loss_div
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # cls_acc
        _, pred_cls = out_cls.max(dim=1)
        corrects = (pred_cls == labels).float()
        acc_cls = corrects.sum()  / batch_size

        # pref_acc
        _, pred_pref = pref_probs.max(dim=1)
        n_pref = (pref != 2).float().sum()
        corrects_pref = (pred_pref == pref)[pref != 2].float()
        acc_pref = corrects_pref.sum() / (1e-8 + n_pref)

        losses['cls'].update(loss_cls.item(), batch_size)
        losses['cls_acc'].update(acc_cls.item(), batch_size)
        losses['pref'].update(loss_pref.item(), batch_size)
        losses['pref_acc'].update(acc_pref.item(), batch_size)
        losses['div'].update(loss_div.item(), 3)
        losses['delta'].update(loss_div.item(), batch_size)

        #
        p_delta = ((probs1 - probs2)[torch.arange(batch_size), labels] >= 0).data
        consistency = (pref == p_delta)[pref != 2].float().mean()
        losses['consistency'].update(consistency.item(), n_pref)

        # save pref
        prefs_all[0, indices] = out_pref1[0].data.cpu()[:, 0]
        prefs_all[1, indices] = out_pref1[1].data.cpu()[:, 0]
        prefs_all[2, indices] = out_pref1[2].data.cpu()[:, 0]
        probs_all[indices] = probs1.data.cpu()

    cos = (saved_probs[0] * saved_probs[1]).sum(dim=-1) / (saved_probs[0].norm(dim=-1) * saved_probs[1].norm(dim=-1))
    cos += (saved_probs[0] * saved_probs[2]).sum(dim=-1) / (saved_probs[0].norm(dim=-1) * saved_probs[2].norm(dim=-1))
    cos += (saved_probs[1] * saved_probs[2]).sum(dim=-1) / (saved_probs[1].norm(dim=-1) * saved_probs[2].norm(dim=-1))
    cos /= 3

    msg = '[Epoch %2d] [AccC %.3f] [LossC %.3f] [AccP %.3f] [LossP %.3f] [LossDel %.3f] [Cos %.3f] [Consist %.3f]' \
          % (epoch, losses['cls_acc'].average, losses['cls'].average, losses['pref_acc'].average,
             losses['pref'].average, losses['delta'].average, cos.mean().item(), losses['consistency'].average)

    if logger:
        logger.log(msg)
    else:
        print(msg)

    return prefs_all, probs_all

def set_loader(args, dataset):
    pair_idx, preference = dataset.train_dataset[:][5], dataset.train_dataset[:][3]
    select_idx_pre = (preference != -1).nonzero()[:, 0]

    select_idx = torch.unique(torch.cat([select_idx_pre, pair_idx[select_idx_pre]]))

    unique_idx = []
    for idx in select_idx:
        if idx not in select_idx_pre:
            unique_idx.append(idx)
    unique_idx = torch.LongTensor(np.array(unique_idx))

    unique_pair = []
    counted = []
    for pre_idx in select_idx_pre:
        if pair_idx[pre_idx] in unique_idx and pair_idx[pre_idx] not in counted:
            unique_pair.append(pre_idx)
            counted.append(pair_idx[pre_idx])
    unique_pair = torch.LongTensor(np.array(unique_pair))
    unique_idx = torch.LongTensor(np.array(counted))

    unique_pref = []
    for pre_idx in unique_pair:
        if preference[pre_idx] == 1:
            unique_pref.append(0)
        elif preference[pre_idx] == 0:
            unique_pref.append(1)
        else:
            unique_pref.append(preference[pre_idx])
    unique_pref = torch.LongTensor(np.array(unique_pref))

    pair_idx[unique_idx] = unique_pair
    preference[unique_idx] = unique_pref

    args.n_samples = len(select_idx)
    
    all_labels = dataset.train_dataset[:][2]
    extended_dataset = TensorDataset(dataset.train_dataset[:][0], dataset.train_dataset[:][0][pair_idx],
                                       all_labels, preference, dataset.train_dataset[:][4])

    pref_train_dataset = TensorDataset(extended_dataset[:][0][select_idx], extended_dataset[:][1][select_idx],
                                       extended_dataset[:][2][select_idx], extended_dataset[:][3][select_idx],
                                       extended_dataset[:][4][select_idx])

    if args.pref_type == 'none':
        train_dataset = TensorDataset(extended_dataset[:][0][select_idx_pre], extended_dataset[:][1][select_idx_pre],
                                       extended_dataset[:][2][select_idx_pre], extended_dataset[:][3][select_idx_pre],
                                       extended_dataset[:][4][select_idx_pre])
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, batch_size=args.batch_size, num_workers=4)
    else:
        print("Number of training samples: {}".format(args.n_samples))

        print("Number of preference A: {}, B: {}, No: {}".format((preference[select_idx] == 1).sum(), (preference[select_idx] == 0).sum(),
                                                                (preference[select_idx] == 2).sum()))
        print("Number of Class Negative: {}, Positive: {}, Neutral: {}".format((pref_train_dataset[:][2] == 0).sum(),
                                                                            (pref_train_dataset[:][2] == 1).sum(),
                                                                            (pref_train_dataset[:][2] == 2).sum()))

        train_loader = DataLoader(pref_train_dataset, shuffle=True, drop_last=False, batch_size=args.batch_size, num_workers=4)
    return train_loader