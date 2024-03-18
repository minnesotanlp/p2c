import os
# import pprint
import json
import copy
import time
import random
import string
import argparse
import numpy as np
from scipy.spatial import distance_matrix

from tqdm import tqdm
from datasets import load_dataset
import openai

def get_cola_query(sentences_a, sentences_b, labels_ab, idx):
    text = \
    f"Read given two sentences A and B, and pick a more {labels_ab[idx]} sentence: \n\
    Sentence A: {sentences_a[idx]}\n\
    Sentence B: {sentences_b[idx]}\n\
    Choices: [Sentence A, Sentence B, No Preference], Answer:"
    
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process GPT-3 API.')
    parser.add_argument('--start', type=int, default=None)
    parser.add_argument('--end', type=int, default=None)
    args = parser.parse_args()

    openai.api_key = "sk-xxx"

    cola_dataset = load_dataset("glue", "cola")
    sentences = cola_dataset['train']['sentence']
    labels = np.array(cola_dataset['train']['label']) 
    label_pools = ['unacceptable (not grammatical)', 'acceptable (grammatical)']

    label_0_indices = list((labels == 0).nonzero()[0])
    label_1_indices = list((labels == 1).nonzero()[0])

    sentences_a0 = [sentences[idx] for idx in label_0_indices]
    selected_rand_idx = np.random.permutation(len(sentences_a0))
    sentences_b0 = [sentences_a0[idx] for idx in selected_rand_idx]
    labels_ab0 = [label_pools[labels[idx]] for idx in label_0_indices]

    sentences_a1 = [sentences[idx] for idx in label_1_indices]
    selected_rand_idx = np.random.permutation(len(sentences_a1))
    sentences_b1 = [sentences_a1[idx] for idx in selected_rand_idx]
    labels_ab1 = [label_pools[labels[idx]] for idx in label_1_indices]
    
    sentences_a = sentences_a0 + sentences_a1
    sentences_b = sentences_b0 + sentences_b1
    labels_ab = labels_ab0 + labels_ab1

    waiting_time = 0.5

    start_idx, end_idx = args.start, args.end
    if start_idx is None and end_idx is None:
        raise ValueError
    elif start_idx is None:
        start_idx = 0
    elif end_idx is None:
        end_idx = len(sentences_a)
    else:
        if start_idx >= end_idx:
            raise ValueError
    
    results_cola = []
    results_idx = []
    for text_idx in tqdm(range(start_idx, end_idx)):
        response_cola = None
        q_cola = get_cola_query(sentences_a, sentences_b, labels_ab, text_idx)
        results_idx.append(idx.tolist())
    
        while response_cola is None:
            try:
                response_cola = openai.Completion.create(
                model="text-davinci-003",
                prompt=q_cola,
                max_tokens=128,
                temperature=0.7
                )
            except:
                time.sleep(waiting_time)
                if waiting_time < 5:
                    waiting_time += 0.5
        
        results_cola.append(response_cola)

    with open(f'./gpt_outputs/results_cola__{start_idx}_{end_idx}.json', "w", encoding='utf-8') as writer:
        writer.write(json.dumps(results_cola, indent=4, ensure_ascii=False) + "\n")