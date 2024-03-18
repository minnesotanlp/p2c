import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data.distributed import DistributedSampler
from collections import defaultdict
import math 
import random
import numpy as np

import pickle as cp
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch import cuda

image_size = 224

# Augmentations
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(image_size, padding=int(image_size / 8)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_val = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class Transform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        return out1

class SUN_Dataset(Dataset):
    def __init__(self, root, txt, transform=None):
        self.img_path = []
        self.labels = []
        self.votes = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
                self.votes.append(float(line.split()[2]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, index

class SUN_Dataset_pref(Dataset):
    def __init__(self, root, txt, pair_idx, pref, transform=transform_train):
        self.img_path = []
        self.labels = []
        self.transform = transform
        with open(txt) as f:
            for line in f:
                self.img_path.append(os.path.join(root, line.split()[0]))
                self.labels.append(int(line.split()[1]))
        
        self.pair_idx = pair_idx
        self.pref = pref

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]

        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        
        path2 = self.img_path[self.pair_idx[index]]

        with open(path2, 'rb') as f:
            sample2 = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample2 = self.transform(sample2)

        return sample, sample2, label, self.pref[index], index

def get_sun_loader(args, pref=False, num_workers=16):
    txt_train = './SUN_train_{}.txt'.format(args.data_type)
    txt_val = './SUN_val_{}.txt'.format(args.data_type)
    txt_test = './SUN_test_{}.txt'.format(args.data_type)

    data_root = args.data_root

    set_train = SUN_Dataset(data_root, txt_train, transform_train)
    set_val = SUN_Dataset(data_root, txt_val, transform_val)
    set_test = SUN_Dataset(data_root, txt_test, transform_val)

    train_loader = DataLoader(set_train, args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=cuda.is_available())
    val_loader = DataLoader(set_val, args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=cuda.is_available())
    test_loader = DataLoader(set_test, args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=cuda.is_available())

    if pref:
        return set_train, train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader, test_loader