import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):

    def __init__(self, params):
        super(Agent, self).__init__()

        self.relation2id_size = len(params['relation2id'])
        self.entity2id_size = len(params['entity2id'])
        self.ePAD = int(params['entity2id']['PAD'])
        self.rPAD = int(params['relation2id']['PAD'])
        self.dummy_start_relation = int(params['relation2id']['DUMMY_START_RELATION'])

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']

        self.batch_size = params['batch_size'] * params['num_rollouts']
                
        self.gwm_model = params.get('gwm_model', None)
        assert self.gwm_model is not None, "GWM Model must be provided for DreamWalker-KG."

        self.gwm_dim = self.gwm_model.config.hidden_dim
        self.embedding_size = self.gwm_dim
        self.hidden_size = params['hidden_size']
        self.num_lstm_layers = params['num_lstm_layers']
        self.action_scoring_chunk_size = max(1, int(params.get('action_scoring_chunk_size', 1)))
        self.embedding_cache_device = str(params.get('embedding_cache_device', 'cpu'))
        self.virtual_edge_tax = float(params.get('virtual_edge_tax', 0.0))

        self._precompute_and_cache_embeddings()

        self.gwm_entity_proj = nn.Linear(self.gwm_dim, 2 * self.embedding_size)
        self.gwm_relation_proj = nn.Linear(self.gwm_dim, 2 * self.embedding_size)

        input_dim = 4 * self.embedding_size
        hidden_dim = 4 * self.hidden_size
        self.policy_step = nn.ModuleList(
            [nn.LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(self.num_lstm_layers)]
        )

        self.policy_hidden = nn.Linear((4 * self.hidden_size) + (4 * self.embedding_size), 4 * self.hidden_size)
        self.policy_output = nn.Linear(4 * self.hidden_size, 4 * self.embedding_size)

    def _precompute_and_cache_embeddings(self):
        """Precomputes and caches all GWM hybrid representations during init."""
        with torch.no_grad():
            self.gwm_model.eval()
            gwm_device = next(self.gwm_model.parameters()).device
            cache_device = torch.device(self.embedding_cache_device)
            
            # Entities
            entity_ids = torch.arange(self.entity2id_size, device=gwm_device)
            cached_ent = self._generate_hybrid_core(entity_ids, kind='entity').to(cache_device)
            self.cached_entity_embeddings = cached_ent
            self.gwm_model.cached_entity_embeddings = self.cached_entity_embeddings
            
            # Relations
            relation_ids = torch.arange(self.relation2id_size, device=gwm_device)
            cached_rel = self._generate_hybrid_core(relation_ids, kind='relation').to(cache_device)
            self.cached_relation_embeddings = cached_rel
            self.gwm_model.cached_relation_embeddings = self.cached_relation_embeddings

            if cache_device.type == 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_hybrid_core(self, ids, kind='entity'):
        pad_id = self.ePAD if kind == 'entity' else self.rPAD
        mask = (ids == pad_id)
        safe_ids = ids.masked_fill(mask, 0)
        
        text_emb = self.gwm_model._lookup_cached_text(safe_ids, kind=kind)
        
        if kind == 'entity':
            struct_emb = self.gwm_model.entity_embeddings(safe_ids)
        else:
            struct_emb = self.gwm_model.relation_embeddings(safe_ids)
            
        fused_emb = self.gwm_model._fuse_modalities(text_emb, struct_emb)
        fused_emb[mask] = 0.0
        return fused_emb

    def _lookup_from_cache(self, cache_tensor, ids, target_device):
        flat_ids = ids.reshape(-1)
        if flat_ids.device != cache_tensor.device:
            flat_ids = flat_ids.to(cache_tensor.device)
        selected = cache_tensor.index_select(0, flat_ids)
        selected = selected.reshape(*ids.shape, -1)
        if selected.device != target_device:
            selected = selected.to(target_device)
        return selected

    def lookup_entity(self, entities):
        emb = self._lookup_from_cache(self.cached_entity_embeddings, entities, self.gwm_entity_proj.weight.device)
        return self.gwm_entity_proj(emb)

    def lookup_relation(self, relations):
        emb = self._lookup_from_cache(self.cached_relation_embeddings, relations, self.gwm_relation_proj.weight.device)
        return self.gwm_relation_proj(emb)

    def get_mem_shape(self):
        return (self.num_lstm_layers, 2, None, 4 * self.hidden_size)

    def init_memory(self, batch_size, device):
        h0 = torch.zeros(self.num_lstm_layers, batch_size, 4 * self.hidden_size, device=device)
        c0 = torch.zeros(self.num_lstm_layers, batch_size, 4 * self.hidden_size, device=device)
        return h0, c0

    def get_dummy_start_label(self, batch_size, device):
        return torch.full((batch_size,), self.dummy_start_relation, dtype=torch.long, device=device)

    def policy_MLP(self, state):
        hidden = F.relu(self.policy_hidden(state))
        output = F.relu(self.policy_output(hidden))
        return output

    def action_encoder(self, next_relations, next_entities):
        relation_embedding = self.lookup_relation(next_relations)
        entity_embedding = self.lookup_entity(next_entities)
        action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def policy_step_forward(self, action_embedding, prev_state):
        prev_h, prev_c = prev_state
        input_t = action_embedding
        next_h = []
        next_c = []
        for layer_idx, cell in enumerate(self.policy_step):
            h_i, c_i = cell(input_t, (prev_h[layer_idx], prev_c[layer_idx]))
            next_h.append(h_i)
            next_c.append(c_i)
            input_t = h_i
        return input_t, (torch.stack(next_h, dim=0), torch.stack(next_c, dim=0))

    def _score_candidate_actions(self, policy_output, next_relations, next_entities):
        """Scores candidate actions in chunks to reduce peak GPU memory."""
        rel_query, ent_query = torch.chunk(policy_output, chunks=2, dim=-1)  # [B, 2E], [B, 2E]
        rel_query = rel_query.unsqueeze(1)
        ent_query = ent_query.unsqueeze(1)
        chunk_scores = []
        num_actions = next_relations.size(1)

        for start in range(0, num_actions, self.action_scoring_chunk_size):
            end = min(start + self.action_scoring_chunk_size, num_actions)
            rel_chunk = next_relations[:, start:end]
            ent_chunk = next_entities[:, start:end]

            # Compute relation/entity contributions sequentially to avoid holding
            # both projected tensors in memory at the same time.
            score_chunk = torch.sum(self.lookup_relation(rel_chunk) * rel_query, dim=2)
            score_chunk = score_chunk + torch.sum(self.lookup_entity(ent_chunk) * ent_query, dim=2)
            chunk_scores.append(score_chunk)

        return torch.cat(chunk_scores, dim=1)

    def step(self, next_relations, next_entities, prev_state, prev_relation,
             query_embedding, current_entities, range_arr, virtual_action_mask=None):
        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        output, new_state = self.policy_step_forward(prev_action_embedding, prev_state)

        prev_entity = self.lookup_entity(current_entities)
        state = torch.cat([output, prev_entity], dim=-1)
        
        state_query_concat = torch.cat([state, query_embedding], dim=-1)

        output = self.policy_MLP(state_query_concat)
        prelim_scores = self._score_candidate_actions(output, next_relations, next_entities)

        # Tax hallucinated (virtual) edges so the policy slightly prefers physical edges.
        if virtual_action_mask is not None and self.virtual_edge_tax > 0.0:
            prelim_scores = prelim_scores - (virtual_action_mask.to(prelim_scores.dtype) * self.virtual_edge_tax)

        mask = next_relations.eq(self.rPAD)
        scores = prelim_scores.masked_fill(mask, -99999.0)

        action = torch.distributions.Categorical(logits=scores).sample().long()

        loss = F.cross_entropy(scores, action, reduction='none')

        action_idx = action
        chosen_relation = next_relations[range_arr, action_idx]

        return loss, new_state, F.log_softmax(scores, dim=1), action_idx, chosen_relation

    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, current_entities,
                 query_relation, range_arr, T=3):

        self.baseline_inputs = []

        query_embedding = self.lookup_relation(query_relation)
        state = self.init_memory(self.batch_size, query_embedding.device)

        prev_relation = self.get_dummy_start_label(self.batch_size, query_embedding.device)

        all_loss = []  # list of loss tensors each [B,]
        all_logits = []  # list of actions each [B,]
        action_idx = []

        for t in range(T):
            next_possible_relations = candidate_relation_sequence[t]
            next_possible_entities = candidate_entity_sequence[t]
            current_entities_t = current_entities[t]

            loss, state, logits, idx, chosen_relation = self.step(
                next_possible_relations,
                next_possible_entities,
                state,
                prev_relation,
                query_embedding,
                current_entities_t,
                range_arr=range_arr
            )

            all_loss.append(loss)
            all_logits.append(logits)
            action_idx.append(idx)
            prev_relation = chosen_relation

        return all_loss, all_logits, action_idx
