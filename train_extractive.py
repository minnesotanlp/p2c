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
from tqdm import tqdm

from datasets import load_dataset
from data_loader import P2CDataset_ext
from model import load_backbone, Classifier, Classifier_multi, Classifier_pref_ensemble
from common import parse_args
from utils import Logger, set_seed, set_model_path, save_model, AverageMeter, ECE

from src.train_ext import set_loader_extractive, train_base_extractive, train_preference_extractive

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    args = parse_args()

    # Set seed
    set_seed(args)

    prefix = f"{args.dataset}_{args.train_type}"
    
    if args.sampling is not None:
        prefix = prefix + "_" + args.sampling
    
    if args.pair_loss:
        log_name = f"{prefix}_pair_cons{args.lambda_cons}_div{args.lambda_div}_S{args.seed}"
    elif args.consistency:
        log_name = f"{prefix}_cons_cons{args.lambda_cons}_div{args.lambda_div}_S{args.seed}"
    else:
        log_name = f"{prefix}_{args.base}_S{args.seed}"

    logger = Logger(log_name)
    log_dir = logger.logdir

    logger.log(args)
    logger.log(log_name)

    logger.log('Loading pre-trained backbone network... ({})'.format(args.backbone))
    backbone, tokenizer = load_backbone(args.backbone)

    logger.log('Initializing model and optimizer...')
    if 'dynasent' in args.dataset or 'mnli' in args.dataset:
        args.n_class = 3
    else:
        args.n_class = 2

    if args.pref_type == 'none':
        if args.base == 'multi':
            model = Classifier_multi(args, args.backbone, backbone, args.n_class, args.train_type).to(device)
        else:    
            model = Classifier(args, args.backbone, backbone, args.n_class, args.train_type).to(device)
    else:    
        model = Classifier_pref_ensemble(args, args.backbone, backbone, args.n_class, args.train_type).to(device)
    
    if args.pre_ckpt is not None:
        logger.log('Loading from pre-trained model')
        model.load_state_dict(torch.load(args.pre_ckpt))

    # Set optimizer (1) fixed learning rate and (2) no weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=0)

    logger.log('Initializing dataset...')
    dataset = P2CDataset_ext(args.dataset, tokenizer, args.backbone)
        
    # Added for preference
    orig_loader = DataLoader(dataset.train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(dataset.val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(dataset.test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

    logger.log('==========> Start training ({})'.format(args.train_type))
    best_acc, final_acc, final_ece = 0, 0, 0

    # Add Tensorboard logger
    train_labels = dataset.train_dataset[:][1]
    pref_train = None
    prob_train = None
    
    train_loader, pair_idx = set_loader_extractive(args, dataset, orig_loader, 1, pref_train, prob_train, train_labels)
    
    for epoch in range(1, 1+args.epochs):
        # Set Dataloader
        if args.pref_type == 'none':
            train_base_extractive(args, train_loader, model, optimizer, epoch, logger)    
        else:
            if epoch > 1:
                train_loader, pair_idx = set_loader_extractive(args, dataset, orig_loader, epoch, pref_train, prob_train, train_labels)
            pref_train, prob_train  = train_preference_extractive(args, train_loader, pair_idx, model, optimizer, epoch, logger)
        best_acc, final_acc, final_ece = eval_func(args, model, val_loader, test_loader, logger, log_dir, epoch,
                                                  best_acc, final_acc, final_ece)

    logger.log('===========>>>>> Final ECE: {}'.format(final_ece))
    logger.log('===========>>>>> Final Test Accuracy: {}'.format(final_acc))

def eval_func(args, model, val_loader, test_loader, logger, log_dir, epoch, best_acc, final_acc, final_ece):
    acc, ece_temp = test_acc(args, val_loader, model, logger)

    if acc > best_acc:
        # As val_data == test_data in GLUE, do not inference it again.
        t_acc, ece = test_acc(args, test_loader, model, logger, ece_temp)

        # Update test accuracy based on validation performance
        best_acc = acc
        final_acc = t_acc
        final_ece = ece

        logger.log('========== Val Acc ==========')
        logger.log('Val acc: {:.3f}'.format(best_acc))
        logger.log('========== Test Acc ==========')
        logger.log('Test acc: {:.3f}'.format(final_acc))
        logger.log('========== Test ECE ==========')
        logger.log('Test ece: {:.3f}'.format(final_ece))

        # Save model
        if args.save_ckpt:
            logger.log('Save model...')
            save_model(args, model, log_dir, epoch)

    return best_acc, final_acc, final_ece

def test_acc(args, loader, model, logger=None, temp_opt=None):
    if logger is not None:
        logger.log('Compute test accuracy...')
    model.eval()

    all_preds = []
    all_labels = []

    for i, (tokens, labels, indices) in enumerate(loader):
        tokens = tokens.long().to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(tokens)

        all_preds.append(outputs)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if temp_opt is None:
        ece = ECE(all_preds, all_labels)
    else:
        ece = ECE(all_preds, all_labels, temp_opt=temp_opt)

    all_preds = all_preds.cpu().max(1)[1]
    all_labels = all_labels.cpu()

    acc = 100.0 * (all_preds == all_labels).float().sum() / len(all_preds)

    return acc, ece

if __name__ == "__main__":
    main()

