from tqdm import tqdm
import json
import numpy as np
from collections import defaultdict
import csv
import random
import os
import torch


class RelationEntityBatcher():
    def __init__(self, input_dir, batch_size, entity2id, relation2id, mode="train"):
        self.input_dir = input_dir
        self.batch_size = batch_size
        
        print('Reading vocab...')
        self.entity2id = entity2id
        self.relation2id = relation2id

        self.mode = mode
        self.create_triples_array()
        print("batcher loaded")

    def get_next_batch(self):
        if self.mode == 'train':
            yield self.yield_next_batch_train()
        else:
            yield self.yield_next_batch_test()

    def create_triples_array(self):
        self.ground_truth_map = defaultdict(set)
        
        # Determine Which Split to Load
        if self.mode == 'train':
            pt_file = os.path.join(self.input_dir, 'train_triples.pt')
        elif self.mode in ['dev', 'valid']:
            pt_file = os.path.join(self.input_dir, 'valid_triples.pt')
            if not os.path.exists(pt_file):
                pt_file = os.path.join(self.input_dir, 'dev_triples.pt')
        else:
            pt_file = os.path.join(self.input_dir, 'test_triples.pt')

        try:
            self.triples_array = torch.load(pt_file).numpy()
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing {pt_file}. Please run preprocess_data.py first.")

        # Build Ground Truth Map
        if self.mode == 'train':
            for e1, r, e2 in self.triples_array:
                self.ground_truth_map[(e1, r)].add(e2)
        else:
            # During eval, the agent cannot be penalized for predicting ANY true fact.
            # We must load all possible facts from all splits into the evaluator store.
            if self.mode in ['dev', 'valid']:
                fact_files = ['train_triples.pt', 'valid_triples.pt', 'dev_triples.pt']
            else:
                fact_files = ['train_triples.pt', 'valid_triples.pt', 'dev_triples.pt', 'test_triples.pt']
            for f in fact_files:
                f_path = os.path.join(self.input_dir, f)
                if os.path.isfile(f_path):
                    f_tensor = torch.load(f_path).numpy()
                    for e1, r, e2 in f_tensor:
                        self.ground_truth_map[(e1, r)].add(e2)

    def yield_next_batch_train(self):
        while True:
            batch_idx = np.random.randint(0, self.triples_array.shape[0], size=self.batch_size)
            batch = self.triples_array[batch_idx, :]
            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.ground_truth_map[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            yield e1, r, e2, all_e2s

    def yield_next_batch_test(self):
        remaining_triples = self.triples_array.shape[0]
        current_idx = 0
        while True:
            if remaining_triples == 0:
                return
            if remaining_triples - self.batch_size > 0:
                batch_idx = np.arange(current_idx, current_idx+self.batch_size)
                current_idx += self.batch_size
                remaining_triples -= self.batch_size
            else:
                batch_idx = np.arange(current_idx, self.triples_array.shape[0])
                remaining_triples = 0
            batch = self.triples_array[batch_idx, :]
            e1 = batch[:,0]
            r = batch[:, 1]
            e2 = batch[:, 2]
            all_e2s = []
            for i in range(e1.shape[0]):
                all_e2s.append(self.ground_truth_map[(e1[i], r[i])])
            assert e1.shape[0] == e2.shape[0] == r.shape[0] == len(all_e2s)
            yield e1, r, e2, all_e2s
