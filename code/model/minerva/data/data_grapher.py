from collections import defaultdict
import logging
import numpy as np
import os
import torch

logger = logging.getLogger(__name__)


class RelationEntityGrapher:
    def __init__(self, data_dir, relation2id, entity2id, max_actions):

        self.ePAD = entity2id['PAD']
        self.rPAD = relation2id['PAD']

        self.data_dir = data_dir
        self.relation2id = relation2id
        self.entity2id = entity2id

        self.adjacency_list = defaultdict(list)
        self.action_space = np.ones((len(entity2id), max_actions, 2), dtype=np.dtype('int32'))
        self.action_space[:, :, 0] *= self.ePAD
        self.action_space[:, :, 1] *= self.rPAD
        self.masked_action_space = None

        self.id2relation = dict([(v, k) for k, v in relation2id.items()])
        self.id2entity = dict([(v, k) for k, v in entity2id.items()])
        self.create_graph()
        print("KG constructed")

    def create_graph(self):
        fact_files = ['train_triples.pt']
        for f in fact_files:
            file_path = os.path.join(self.data_dir, f)
            if os.path.exists(file_path):
                tensor_data = torch.load(file_path).numpy()
                for e1, r, e2 in tensor_data:
                    self.adjacency_list[e1].append((r, e2))

        for e1 in self.adjacency_list:
            num_actions = 1
            self.action_space[e1, 0, 1] = self.relation2id['NO_OP']
            self.action_space[e1, 0, 0] = e1
            for r, e2 in self.adjacency_list[e1]:
                if num_actions == self.action_space.shape[1]:
                    break
                self.action_space[e1,num_actions,0] = e2
                self.action_space[e1,num_actions,1] = r
                num_actions += 1
        del self.adjacency_list
        self.adjacency_list = None

    def return_next_actions(self, current_entities, start_entities, query_relations, answers, all_correct_answers, last_step, rollouts, gwm_model=None, k=3):
        ret = self.action_space[current_entities, :, :].copy()
        
        if gwm_model is not None and hasattr(gwm_model, 'predict_latent_jumps'):
            batch_sz = current_entities.shape[0]
            virtual_edges = np.zeros((batch_sz, k, 2), dtype=np.dtype('int32'))
            
            try:
                virtual_entities = gwm_model.predict_latent_jumps(current_entities, query_relations, k=k)
            except Exception as e:
                logger.warning(f"GWM prediction failed: {e}")
                virtual_entities = None

            if virtual_entities is not None:
                virtual_entities = np.asarray(virtual_entities, dtype=np.int32)
                if virtual_entities.shape == (batch_sz, k):
                    valid_mask = np.logical_and(virtual_entities >= 0, virtual_entities < len(self.entity2id))
                    safe_virtual_entities = np.where(valid_mask, virtual_entities, self.ePAD)

                    virtual_edges[:, :, 0] = safe_virtual_entities
                    virtual_edges[:, :, 1] = np.expand_dims(query_relations, axis=1) # Hallucinated jump follows query relation

                    # Concatenate virtual edges to the physical edges
                    ret = np.concatenate([ret, virtual_edges], axis=1)
                else:
                    logger.warning(
                        f"GWM predict_latent_jumps returned shape {virtual_entities.shape}; expected {(batch_sz, k)}. Skipping hallucinated edges."
                    )

        for i in range(current_entities.shape[0]):
            if current_entities[i] == start_entities[i]:
                relations = ret[i, :, 1]
                entities = ret[i, :, 0]
                mask = np.logical_and(relations == query_relations[i] , entities == answers[i])
                ret[i, :, 0][mask] = self.ePAD
                ret[i, :, 1][mask] = self.rPAD
            if last_step:
                entities = ret[i, :, 0]
                relations = ret[i, :, 1]

                correct_e2 = answers[i]
                for j in range(entities.shape[0]):
                    if entities[j] in all_correct_answers[i // rollouts] and entities[j] != correct_e2:
                        entities[j] = self.ePAD
                        relations[j] = self.rPAD

        return ret
