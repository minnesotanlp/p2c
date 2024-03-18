#!/usr/bin/env python3 -u

from __future__ import print_function

import argparse
import csv
import os
import json
import copy

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, TensorDataset

from torchvision import models
from torch import cuda

import data_loader

from argument import parser, print_args
from utils_bef import progress_bar, checkpoint, one_hot, get_highest_incorrect_predict, adjust_learning_rate
from collections import OrderedDict
from model_loader import Classifier_pref_ensemble

args = parser()
print_args(args)

use_cuda = torch.cuda.is_available()

best_val = 0 # best validation accuracy
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
num_class = 2
if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')

dataset, trainloader, valloader, testloader = data_loader.get_sun_loader(args, pref=True)
# Model
print('==> Building model..')
model = Classifier_pref_ensemble(args)
model = model.cuda()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

if not os.path.isdir('results'):
    os.mkdir('results')

args.name = 'SUN' + str(args.data_type) + '_' + args.train_type + '_' + args.model

loginfo = 'results/log_' + args.name + '_' + str(args.seed)
logname = (loginfo+ '.csv')

print('Training info...')
print(loginfo)

if use_cuda:
    model.cuda()
    print(torch.cuda.device_count())
    cudnn.benchmark = True
    print('Using CUDA..')

def set_loader(args, dataset, orig_loader, epoch):
    pair_idx_pref = torch.LongTensor(np.load('./dataset/SUN_{}_idx_pref_random20.npy'.format(args.data_type)))
    args.n_samples = pair_idx_pref.size()[1]

    txt_train = './dataset/SUN_train_{}.txt'.format(args.data_type)
    data_root = args.data_root

    if args.static or (epoch == 0):
        pair_idx, preference = pair_idx_pref[0, :, 0], pair_idx_pref[0, :, 1]
    else:
        pair_idx, preference = pair_idx_pref[int(epoch/10), :, 0], pair_idx_pref[int(epoch/10), :, 1]
    pref_train_dataset = data_loader.SUN_Dataset_pref(data_root, txt_train, pair_idx, preference)

    trainloader = DataLoader(pref_train_dataset, args.batch_size, shuffle=True, num_workers=16, pin_memory=cuda.is_available())

    return trainloader, pair_idx

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

def train(epoch, criterion):
    print('\nEpoch: %d' % epoch)

    model.train()

    train_loss = 0
    pref_loss = 0
    correct = 0
    total = 0
    correct_pref = 0
    total_pref = 0
    
    steps = 0

    soft_labels = torch.Tensor(np.load('./dataset/SUN_{}_soft_labels.npy'.format(args.data_type))).cuda()

    for batch_idx, (inputs1, inputs2, labels, pref, indices) in enumerate(trainloader):
        batch_size = inputs1.size(0)
        
        inputs1, inputs2, labels, pref = inputs1.cuda(), inputs2.cuda(), labels.cuda(), pref.cuda()
        steps += 1

        pref1, pref2 = pref.clone(), pref.clone()
        pref1[pref1 == 2] = 0
        pref2[pref2 == 2] = 1
        pref_label = torch.zeros(batch_size, 2).cuda()
        pref_label[torch.arange(batch_size), pref1] += 0.5
        pref_label[torch.arange(batch_size), pref2] += 0.5

        out_cls, out_pref1 = model(inputs1, y=labels, pref=True)
        out_cls2, out_pref2 = model(inputs2, y=labels, pref=True)
        
        loss_cls = criterion(out_cls, labels).mean()

        probs1, probs2 = out_cls.softmax(dim=-1), out_cls2.softmax(dim=-1)
        zeros = torch.zeros(batch_size).float().cuda()

        if args.pair_loss:
            soft_labels_batch, soft_labels_batch2 = soft_labels[indices], soft_labels[pair_idx[indices]]

            # Delta
            soft_labels_delta, prob_delta = (soft_labels_batch - soft_labels_batch2), (probs1 - probs2)
            mask1, mask2 = (soft_labels_delta >= 0).float(), (soft_labels_delta < 0).float()
            loss_cons = (mask1 * torch.max(zeros.unsqueeze(1), soft_labels_delta - prob_delta)).sum(dim=-1)
            loss_cons += (mask2 * torch.max(zeros.unsqueeze(1), prob_delta - soft_labels_delta)).sum(dim=-1)
            loss_cons = loss_cons.mean()
        else:
            loss_cons = torch.Tensor([0]).cuda().mean()

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

        loss = args.lambda_cls * loss_cls + args.lambda_cons * loss_cons + args.lambda_pref * loss_pref - args.lambda_div  * loss_div

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss.data
        pref_loss += loss_pref.data

        # cls_acc
        _, predicted = out_cls.max(dim=1)
        total += labels.size(0)
        correct += predicted.eq(labels.data).cpu().sum().float()

        # pref_acc
        _, pred_pref = pref_probs.max(dim=1)
        n_pref = (pref != 2).float().sum()
        corrects_pref = (pred_pref == pref)[pref != 2].float()
        total_pref += n_pref
        correct_pref += corrects_pref.cpu().sum()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Loss_pref: %.3f | Acc: %.2f%% (%d/%d) | Acc Pref: %.2f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), pref_loss/(batch_idx+1),
                        100.*correct/float(total), correct, total,
                        100.*correct_pref/float(total_pref), correct_pref, total_pref) )

    return (train_loss/batch_idx, 100.*correct/float(total))

def val(epoch, criterion):
    global best_val
    model.eval()
    class_acc = torch.zeros(num_class)

    val_loss = 0
    correct = 0
    total = 0
    batch_idx = 0

    for batch_idx, (inputs, targets, _) in enumerate(valloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        val_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx == 0:
            total_preds = predicted.data.cpu()
            total_targets = targets.data.cpu()
        else:
            total_preds = torch.cat([total_preds, predicted.data.cpu()], 0)
            total_targets = torch.cat([total_targets, targets.data.cpu()], 0)

        progress_bar(batch_idx, len(valloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (val_loss / (batch_idx + 1), 100. * correct / (float(total)),
                        correct, total))

    for i in range(num_class):
        class_mask = (total_targets == i).float()
        pred_mask = (total_preds == i).float()
        class_acc[i] += (class_mask * pred_mask).sum()
        class_acc[i] /= (class_mask).sum()

    acc = 100. * correct / total
    print("Validation accuracy: {0}".format(acc))
    
    if class_acc.mean() > best_val:
        best_val = class_acc.mean()
        test_loss, test_acc, test_mean_acc = test(epoch, criterion)        

        # Save the checkpoint
        checkpoint(model, test_acc, epoch, args, optimizer)
        
        return test_loss, test_acc, test_mean_acc
    else:
        return torch.Tensor([0]), torch.Tensor([0]), torch.Tensor([0]) 

def test(epoch, criterion):
    global best_acc
    model.eval()
    class_acc = torch.zeros(num_class)

    test_loss = 0
    correct = 0
    total = 0
    batch_idx = 0

    for batch_idx, (inputs, targets, _) in enumerate(testloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        
        with torch.no_grad():
            outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx == 0:
            total_preds = predicted.data.cpu()
            total_targets = targets.data.cpu()
        else:
            total_preds = torch.cat([total_preds, predicted.data.cpu()], 0)
            total_targets = torch.cat([total_targets, targets.data.cpu()], 0)

        progress_bar(batch_idx, len(testloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / (float(total)),
                        correct, total))

    for i in range(num_class):
        class_mask = (total_targets == i).float()
        pred_mask = (total_preds == i).float()
        class_acc[i] += (class_mask * pred_mask).sum()
        class_acc[i] /= (class_mask).sum()

    acc = 100. * correct / total
    print("Test accuracy: {0}".format(acc))
    
    best_acc = class_acc.mean()
        
    return (test_loss / batch_idx, 100. * correct / float(total), class_acc.mean() * 100)

##### Log file for training selected tasks #####
if os.path.exists(logname):
    os.remove(logname) 

with open(logname, 'w') as logfile:
    logwriter = csv.writer(logfile, delimiter=',')
    logwriter.writerow(['epoch', 'train loss', 'train acc',
                        'test loss', 'test acc'])

criterion = nn.CrossEntropyLoss().cuda()
trainloader, pair_idx = set_loader(args, dataset, trainloader, 0)

for epoch in range(start_epoch, args.epoch):
    adjust_learning_rate(optimizer, epoch, args)
    
    train_loss, train_acc = train(epoch, criterion)
    test_loss, test_acc, test_mean_acc = val(epoch, criterion)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss.item(), train_acc.item(), test_loss.item(), test_acc.item()])

print("Final Test accuracy: {0}".format(best_acc))
        
