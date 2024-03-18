import os
import json
from abc import *

import torch
import csv
from torch.utils.data import TensorDataset
import numpy as np
import pandas as pd
import gzip
import numpy as np
import logging
from datasets import load_dataset
DATA_PATH = './datasets'

def load_jsonl_fast(txt):
    results = []
    lines = [t for t in txt.split('\n') if t.strip()!='']
        
    if len(lines) > 0:
        for ln, line in enumerate(lines):
            results.append(json.loads(line))
        return results
    else:
        return False

def create_tensor_dataset(inputs, labels, index):
    assert len(inputs) == len(labels)
    assert len(inputs) == len(index)

    inputs = torch.stack(inputs)  # (N, T)
    labels = torch.stack(labels)
    index = np.array(index)
    index = torch.Tensor(index).long()

    dataset = TensorDataset(inputs, labels, index)

    return dataset

def create_tensor_dataset_pref(inputs, labels, index, pair_inputs, pref_labels, pref_indices=None):
    assert len(inputs) == len(labels)
    assert len(inputs) == len(index)

    inputs = torch.stack(inputs)  # (N, T)
    labels = torch.stack(labels)
    index = np.array(index)
    index = torch.Tensor(index).long()

    pair_inputs = torch.stack(pair_inputs)  # (N, T)
    pref_labels = torch.stack(pref_labels)

    if pref_indices is None:
        dataset = TensorDataset(inputs, pair_inputs, labels, pref_labels, index)
    else:
        pref_indices = np.array(pref_indices)
        pref_indices = torch.Tensor(pref_indices).long()
        dataset = TensorDataset(inputs, pair_inputs, labels, pref_labels, index, pref_indices)        

    return dataset

class BaseDataset(metaclass=ABCMeta):
    def __init__(self, data_name, tokenizer, backbone='roberta', seed=0):

        self.data_name = data_name
        self.tokenizer = tokenizer
        self.seed = seed
        self.backbone = backbone

        if not self._check_exists():
            self._preprocess()

        print(backbone)

        self.train_dataset = torch.load(self._train_path)
        self.val_dataset = torch.load(self._val_path)
        self.test_dataset = torch.load(self._test_path)

    @property
    def _train_path(self):
        return os.path.join(DATA_PATH, self.data_name + '_' + self.backbone +'_train.pth')

    @property
    def _val_path(self):
        return os.path.join(DATA_PATH, self.data_name + '_' + self.backbone +'_val.pth')

    @property
    def _test_path(self):
        return os.path.join(DATA_PATH, self.data_name + '_' + self.backbone +'_test.pth')

    def _check_exists(self):
        if not os.path.exists(self._train_path):
            return False
        elif not os.path.exists(self._val_path):
            return False
        elif not os.path.exists(self._test_path):
            return False
        else:
            return True

    @abstractmethod
    def _preprocess(self):
        pass

    @abstractmethod
    def _load_dataset(self, *args, **kwargs):
        pass

class P2CDataset(BaseDataset):
    def __init__(self, data_name, tokenizer, seed=0):
        super(P2CDataset, self).__init__(data_name, tokenizer, seed)

        self.data_name = data_name

    def _preprocess(self):
        print('Pre-processing {} dataset...'.format(self.data_name))
        train_dataset = self._load_dataset('train')
        if self.data_name == 'cola':
            val_dataset = self._load_dataset('test')
            test_dataset = val_dataset
        else:
            val_dataset = self._load_dataset('validation')
            test_dataset = self._load_dataset('test')

        # Use the same dataset for validation and test
        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'validation', 'test']

        if self.data_name == 'cola':
            data_set = load_dataset('JaehyungKim/p2c_cola')[mode]
        elif self.data_name == 'emo': 
            data_set = load_dataset('JaehyungKim/p2c_emo')[mode]
        elif self.data_name == 'hate': 
            data_set = load_dataset('JaehyungKim/p2c_hate')[mode]
        elif self.data_name == 'spam': 
            data_set = load_dataset('JaehyungKim/p2c_spam')[mode]
        elif 'dynasent2_sub' in self.data_name: 
            data_set = load_dataset('JaehyungKim/p2c_dynasent2_all')[mode]

        # Get the lists of sentences and their labels.
        inputs, labels, indices = [], [], []
        pair_inputs, pref_labels = [], []
        pair_indices = []

        for i in range(len(data_set)):
            data_n = data_set[i]

            if self.data_name == 'cola':
                max_len = 128
            else:
                max_len = 256
            toks = self.tokenizer.encode(data_n['sentence'], add_special_tokens=True, max_length=max_len, pad_to_max_length=True,
                                return_tensors='pt')[0]
            label = torch.tensor(data_n['label']).long()
            
            inputs.append(toks)
            labels.append(label)
            indices.append(i)

            if mode == 'train':
                pair_toks = self.tokenizer.encode(data_n['pair_sentence'], add_special_tokens=True, max_length=256, pad_to_max_length=True,
                                        return_tensors='pt')[0]
                if 'dynasent2_sub' in self.data_name:
                    if 'generative' in self.data_name:
                        pref_label = torch.tensor(data_n['generative_preference_label']).long()
                    elif 'extractive' in self.data_name:
                        pref_label = torch.tensor(data_n['extractive_preference_label']).long()
                    elif 'subjective' in self.data_name:
                        pref_label = torch.tensor(data_n['subjective_preference_label']).long()
                    else:
                        raise ValueError("Wrong type of preference label")
                    pair_indices.append(data_n['pair_sentence_idx'])
                else:
                    pref_label = torch.tensor(data_n['preference_label']).long()
                pair_inputs.append(pair_toks)
                pref_labels.append(pref_label)
        if mode == 'train':
            if 'dynasent2_sub' in self.data_name:
                dataset = create_tensor_dataset_pref(inputs, labels, indices, pair_inputs, pref_labels, pair_indices)
            else:
                dataset = create_tensor_dataset_pref(inputs, labels, indices, pair_inputs, pref_labels)
        else:
            dataset = create_tensor_dataset(inputs, labels, indices)

        return dataset

class P2CDataset_ext(BaseDataset):
    def __init__(self, data_name, tokenizer, seed=0):
        super(P2CDataset_ext, self).__init__(data_name, tokenizer, seed)

        self.data_name = data_name

    def _preprocess(self):
        print('Pre-processing {} dataset...'.format(self.data_name))
        train_dataset = self._load_dataset('train')
        val_dataset = self._load_dataset('validation')
        test_dataset = self._load_dataset('test')

        # Use the same dataset for validation and test
        torch.save(train_dataset, self._train_path)
        torch.save(val_dataset, self._val_path)
        torch.save(test_dataset, self._test_path)

    def _load_dataset(self, mode='train', raw_text=False):
        assert mode in ['train', 'validation', 'test']

        if self.data_name == 'dynasent1':
            data_set = load_dataset('JaehyungKim/p2c_dynasent1')[mode]
        elif self.data_name == 'dynasent2': 
            data_set = load_dataset('JaehyungKim/p2c_dynasent2')[mode]
        elif self.data_name == 'mnli': 
            data_set = load_dataset('JaehyungKim/p2c_mnli')[mode]
        elif self.data_name == 'offensive': 
            data_set = load_dataset('JaehyungKim/p2c_offensive')[mode]
        elif self.data_name == 'polite_stack': 
            data_set = load_dataset('JaehyungKim/p2c_polite_stack')[mode]
        elif self.data_name == 'polite_wiki': 
            data_set = load_dataset('JaehyungKim/p2c_polite_wiki')[mode]
            
        # Get the lists of sentences and their labels.
        inputs, labels, indices = [], [], []

        for i in range(len(data_set)):
            data_n = data_set[i]

            toks = self.tokenizer.encode(data_n['sentence'], add_special_tokens=True, max_length=256, pad_to_max_length=True,
                                return_tensors='pt')[0]
            label = torch.tensor(data_n['label']).long()
            
            inputs.append(toks)
            labels.append(label)
            indices.append(i)
            
        dataset = create_tensor_dataset(inputs, labels, indices)

        return dataset
