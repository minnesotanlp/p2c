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
from data_loader import P2CDataset
from model import load_backbone, Classifier, Classifier_pref_ensemble
from common import parse_args
from utils import Logger, set_seed, set_model_path, save_model, AverageMeter, cut_input, ECE
from src.train import train_base, train_preference, set_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = './checkpoint'

def main():
    args = parse_args()

    # Set seed
    set_seed(args)

    if args.pref_type == 'none':
        log_name = f"{args.dataset}_{args.train_type}_{args.base}_S{args.seed}"
    else:
        log_name = f"{args.dataset}_{args.train_type}_{args.pref_type}_cons{args.lambda_cons}_div{args.lambda_div}_S{args.seed}"

    logger = Logger(log_name)
    log_dir = logger.logdir

    logger.log(args)
    logger.log(log_name)

    logger.log('Loading pre-trained backbone network... ({})'.format(args.backbone))
    backbone, tokenizer = load_backbone(args.backbone)

    logger.log('Initializing model and optimizer...')
    if 'dynasent' in args.dataset:
        args.n_class = 3
    else:
        args.n_class = 2

    if args.pref_type == 'none':
        model = Classifier(args, args.backbone, backbone, args.n_class, args.train_type).to(device)
    else:    
        model = Classifier_pref_ensemble(args, args.backbone, backbone, args.n_class, args.train_type).to(device)
    
    if args.pre_ckpt is not None:
        logger.log('Loading from pre-trained model')
        model.load_state_dict(torch.load(args.pre_ckpt))

    # Set optimizer (1) fixed learning rate and (2) no weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.model_lr, weight_decay=0)

    logger.log('Initializing dataset...')
    dataset = P2CDataset(args.dataset, tokenizer, args.backbone)
        
    # Added for preference
    orig_loader = DataLoader(dataset.train_dataset, shuffle=True, drop_last=True, batch_size=args.batch_size, num_workers=4)
    val_loader = DataLoader(dataset.val_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)
    test_loader = DataLoader(dataset.test_dataset, shuffle=False, batch_size=args.batch_size, num_workers=4)

    logger.log('==========> Start training ({})'.format(args.train_type))
    best_acc, final_acc, final_ece, final_group = 0, 0, 0, 0

    # Construction of dataloader
    train_loader = set_loader(args, dataset)

    for epoch in range(1, 1+args.epochs):
        # Set Dataloader
        if args.pref_type == 'none':
            train_base(args, train_loader, model, optimizer, epoch, logger)
        else:
            pref_train, prob_train = train_preference(args, train_loader, model, optimizer, epoch, logger)

        best_acc, final_acc, final_ece, final_group = eval_func(args, model, val_loader, test_loader, logger, log_dir, epoch,
                                                  best_acc, final_acc, final_ece, final_group)

    logger.log('===========>>>>> Final ECE: {}'.format(final_ece))
    logger.log('===========>>>>> Final Test Accuracy: {}'.format(final_acc))
    logger.log('===========>>>>> Final Group:Disagree: {:.3f}, Neutral: {:.3f}, Agreed: {:.3f}'.format(final_group[0],
                                                                                                       final_group[1],
                                                                                                       final_group[2]))

def eval_func(args, model, val_loader, test_loader, logger, log_dir, epoch, best_acc, final_acc, final_ece, final_group):
    acc, ece_temp = test_acc(args, val_loader, model, logger)

    if acc > best_acc:
        t_acc, ece, group_acc = test_acc(args, test_loader, model, logger, ece_temp, test=True)

        # Update test accuracy based on validation performance
        best_acc = acc
        final_acc = t_acc
        final_ece = ece
        final_group = group_acc

        logger.log('========== Val Acc ==========')
        logger.log('Val acc: {:.3f}'.format(best_acc))
        logger.log('========== Test Acc ==========')
        logger.log('Test acc: {:.3f}'.format(final_acc))
        logger.log('========== Test ECE ==========')
        logger.log('Test ece: {:.3f}'.format(final_ece))
        logger.log('========== Test Group Acc ==========')
        logger.log('Disagree: {:.3f}, Neutral: {:.3f}, Agreed: {:.3f}'.format(final_group[0], final_group[1], final_group[2]))

        # Save model
        if args.save_ckpt:
            logger.log('Save model...')
            save_model(args, model, log_dir, epoch)

    return best_acc, final_acc, final_ece, final_group

def test_acc(args, loader, model, logger=None, temp_opt=None, test=False):
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

    if test:
        soft_labels = torch.Tensor(np.load(f'./{args.pre_gen}/dynasent2_test_soft_label.npy'))
        values = soft_labels.max(dim=-1)[0].unique().unique()

        group_acc = torch.zeros(len(values))
        for i in range(len(values)):
            i_idx = (soft_labels.max(dim=-1)[0] == values[i])
            group_acc[i] = 100.0 * (all_preds[i_idx] == all_labels[i_idx]).float().sum() / len(all_preds[i_idx])

        return acc, ece, group_acc
    else:
        return acc, ece

if __name__ == "__main__":
    main()

