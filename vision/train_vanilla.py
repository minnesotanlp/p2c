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
from torchvision import models

import data_loader

from argument import parser, print_args
from utils_bef import progress_bar, checkpoint, one_hot, get_highest_incorrect_predict, adjust_learning_rate
from collections import OrderedDict

args = parser()
print_args(args)

use_cuda = torch.cuda.is_available()

best_val = 0 # best validation accuracy
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.seed != 0:
    torch.manual_seed(args.seed)

# Data
print('==> Preparing data..')

trainloader, valloader, testloader = data_loader.get_sun_loader(args)
# Model
print('==> Building model..')
model = models.resnet18(pretrained=False)
num_class = 2
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_class)
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

def set_grad(var):
    def hook(grad):
        var.grad = grad
    return hook

def train(epoch, criterion):
    print('\nEpoch: %d' % epoch)

    model.train()

    train_loss = 0
    correct = 0
    total = 0
    steps = 0

    for batch_idx, (inputs, targets, indices) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        steps += 1

        outputs = model(inputs)

        loss = criterion(outputs, targets)
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum().float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Total Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1),
                        100.*correct/float(total), correct, total) )

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

for epoch in range(start_epoch, args.epoch):
    adjust_learning_rate(optimizer, epoch, args)

    train_loss, train_acc = train(epoch, criterion)
    test_loss, test_acc, test_mean_acc = val(epoch, criterion)

    with open(logname, 'a') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([epoch, train_loss.item(), train_acc.item(), test_loss.item(), test_acc.item()])

print("Final Test accuracy: {0}".format(best_acc))
        
