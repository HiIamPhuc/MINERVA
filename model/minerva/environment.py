from __future__ import absolute_import
from __future__ import division
import numpy as np
from data_batcher import RelationEntityBatcher
from data_grapher import RelationEntityGrapher
import logging


class Episode(object):

    def __init__(self, graph, data, params):
        self.grapher = graph
        self.batch_size, \
            self.path_len, num_rollouts, test_rollouts, \
            positive_reward, negative_reward, \
            mode, batcher, \
            self.gwm_model, self.hallucinate_k = params
        
        self.mode = mode
        self.num_rollouts = num_rollouts if self.mode == 'train' else test_rollouts
        self.current_hop = 0
        start_entities, query_relation,  end_entities, all_answers = data
        self.no_examples = start_entities.shape[0]
        self.positive_reward = positive_reward
        self.negative_reward = negative_reward
        self.start_entities = np.repeat(start_entities, self.num_rollouts)
        self.query_relation = np.repeat(query_relation, self.num_rollouts)
        self.end_entities = np.repeat(end_entities, self.num_rollouts)
        self.current_entities = np.array(self.start_entities)
        self.all_answers = all_answers

        self.state = {}
        self._update_state()

    def _update_state(self):
        next_actions = self.grapher.return_next_actions(self.current_entities, self.start_entities, self.query_relation,
                                                        self.end_entities, self.all_answers, self.current_hop == self.path_len - 1,
                                                        self.num_rollouts, self.gwm_model, self.hallucinate_k)
        self.state['next_relations'] = next_actions[:, :, 1]
        self.state['next_entities'] = next_actions[:, :, 0]
        self.state['current_entities'] = self.current_entities

    def get_state(self):
        return self.state

    def get_query_relation(self):
        return self.query_relation

    def get_reward(self):
        reward = (self.current_entities == self.end_entities)
        base_reward = np.where(reward, self.positive_reward, self.negative_reward).astype(np.float32)

        # --- GWM-RNN SOFT REWARD INTEGRATION ---
        if self.gwm_model is not None:
            import torch
            false_mask = (reward == False)
            if np.any(false_mask):
                curr_ent_t = torch.tensor(self.current_entities, dtype=torch.long)
                end_ent_t = torch.tensor(self.end_entities, dtype=torch.long)
                
                # Prevent ePAD lookups
                ePAD = int(self.grapher.entity2id['PAD'])
                mask_curr = (curr_ent_t == ePAD)
                mask_end = (end_ent_t == ePAD)
                curr_ent_safe = curr_ent_t.masked_fill(mask_curr, 0)
                end_ent_safe = end_ent_t.masked_fill(mask_end, 0)
                
                with torch.no_grad():
                    device = next(self.gwm_model.parameters()).device

                    cache = self.gwm_model.cached_entity_embeddings
                    cache_device = cache.device
                    curr_ids = curr_ent_safe.to(cache_device)
                    end_ids = end_ent_safe.to(cache_device)
                    curr_emb = cache.index_select(0, curr_ids.reshape(-1)).reshape(*curr_ids.shape, -1).to(device)
                    end_emb = cache.index_select(0, end_ids.reshape(-1)).reshape(*end_ids.shape, -1).to(device)
                    
                    # Cosine Similarity
                    curr_emb = torch.nn.functional.normalize(curr_emb, p=2, dim=-1)
                    end_emb = torch.nn.functional.normalize(end_emb, p=2, dim=-1)
                    
                    sim = (curr_emb * end_emb).sum(dim=-1).cpu().numpy()
                    
                    # Bound soft reward to positive similarities between 0.0 and 1.0
                    soft_reward = np.maximum(0, sim)
                    
                    # Zero out padding elements
                    soft_reward[mask_curr.numpy()] = self.negative_reward
                    soft_reward[mask_end.numpy()] = self.negative_reward
                    
                    # Apply soft reward to false matches
                    base_reward[false_mask] = soft_reward[false_mask]

        return base_reward

    def __call__(self, action):
        self.current_hop += 1
        self.current_entities = self.state['next_entities'][np.arange(self.no_examples*self.num_rollouts), action]
        self._update_state()
        return self.state


class Environment(object):
    def __init__(self, params, mode='train'):

        self.batch_size = params['batch_size']
        self.num_rollouts = params['num_rollouts']
        self.positive_reward = params['positive_reward']
        self.negative_reward = params['negative_reward']
        self.mode = mode
        self.path_len = params['path_length']
        self.test_rollouts = params['test_rollouts']

        self.batcher = RelationEntityBatcher(input_dir=params['data_dir'],
                                             mode=self.mode,
                                             batch_size=params['batch_size'],
                                             entity2id=params['entity2id'],
                                             relation2id=params['relation2id'])
        if self.mode != 'train':
            self.total_no_examples = self.batcher.triples_array.shape[0]
            
        self.grapher = RelationEntityGrapher(data_dir=params['data_dir'],
                                             max_actions=params['max_actions'],
                                             entity2id=params['entity2id'],
                                             relation2id=params['relation2id'])
        
        self.gwm_model = params.get('gwm_model', None)
        assert self.gwm_model is not None, "GWM Model must be provided for DreamWalker-KG."
        self.hallucinate_k = params.get('hallucinate_k', 3)

    def get_episodes(self):
        params = self.batch_size, self.path_len, self.num_rollouts, self.test_rollouts, self.positive_reward, self.negative_reward, self.mode, self.batcher, self.gwm_model, self.hallucinate_k
        if self.mode == 'train':
            for data in self.batcher.yield_next_batch_train():

                yield Episode(self.grapher, data, params)
        else:
            for data in self.batcher.yield_next_batch_test():
                if data == None:
                    return
                yield Episode(self.grapher, data, params)
