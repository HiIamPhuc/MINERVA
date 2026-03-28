import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):

    def __init__(self, params):
        super(Agent, self).__init__()

        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.ePAD = int(params['entity_vocab']['PAD'])
        self.rPAD = int(params['relation_vocab']['PAD'])
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']

        self.num_rollouts = params['num_rollouts']
        self.test_rollouts = params['test_rollouts']
        self.LSTM_Layers = params['LSTM_layers']
        self.batch_size = params['batch_size'] * params['num_rollouts']
        self.dummy_start_relation = int(params['relation_vocab']['DUMMY_START_RELATION'])

        self.entity_embedding_size = self.embedding_size
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.m = 4
        else:
            self.m = 2

        self.relation_lookup_table = nn.Embedding(self.action_vocab_size, 2 * self.embedding_size)
        nn.init.xavier_uniform_(self.relation_lookup_table.weight)
        self.relation_lookup_table.weight.requires_grad = self.train_relations

        self.entity_lookup_table = nn.Embedding(self.entity_vocab_size, 2 * self.entity_embedding_size)
        if params['use_entity_embeddings']:
            nn.init.xavier_uniform_(self.entity_lookup_table.weight)
        else:
            with torch.no_grad():
                self.entity_lookup_table.weight.zero_()
        self.entity_lookup_table.weight.requires_grad = self.train_entities

        input_dim = self.m * self.embedding_size
        hidden_dim = self.m * self.hidden_size
        self.policy_step = nn.ModuleList(
            [nn.LSTMCell(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(self.LSTM_Layers)]
        )

        self.policy_hidden = nn.Linear((self.m * self.hidden_size) + (2 * self.embedding_size), 4 * self.hidden_size)
        self.policy_output = nn.Linear(4 * self.hidden_size, self.m * self.embedding_size)

    def get_mem_shape(self):
        return (self.LSTM_Layers, 2, None, self.m * self.hidden_size)

    def init_memory(self, batch_size, device):
        h0 = torch.zeros(self.LSTM_Layers, batch_size, self.m * self.hidden_size, device=device)
        c0 = torch.zeros(self.LSTM_Layers, batch_size, self.m * self.hidden_size, device=device)
        return h0, c0

    def get_dummy_start_label(self, batch_size, device):
        return torch.full((batch_size,), self.dummy_start_relation, dtype=torch.long, device=device)

    def policy_MLP(self, state):
        hidden = F.relu(self.policy_hidden(state))
        output = F.relu(self.policy_output(hidden))
        return output

    def action_encoder(self, next_relations, next_entities):
        relation_embedding = self.relation_lookup_table(next_relations)
        entity_embedding = self.entity_lookup_table(next_entities)
        if self.use_entity_embeddings:
            action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        else:
            action_embedding = relation_embedding
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

    def step(self, next_relations, next_entities, prev_state, prev_relation, query_embedding, current_entities,
             label_action, range_arr, first_step_of_test):

        prev_action_embedding = self.action_encoder(prev_relation, current_entities)
        output, new_state = self.policy_step_forward(prev_action_embedding, prev_state)

        # Get state vector
        prev_entity = self.entity_lookup_table(current_entities)
        if self.use_entity_embeddings:
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            state = output
        candidate_action_embeddings = self.action_encoder(next_relations, next_entities)
        state_query_concat = torch.cat([state, query_embedding], dim=-1)

        # MLP for policy#

        output = self.policy_MLP(state_query_concat)
        output_expanded = output.unsqueeze(1)  # [B, 1, mE]
        prelim_scores = torch.sum(candidate_action_embeddings * output_expanded, dim=2)

        # Masking PAD actions

        mask = next_relations.eq(self.rPAD)
        scores = prelim_scores.masked_fill(mask, -99999.0)

        # 4 sample action
        action = torch.distributions.Categorical(logits=scores).sample().long()

        # loss
        # 5a.
        loss = F.cross_entropy(scores, action, reduction='none')

        # 6. Map back to true id
        action_idx = action
        chosen_relation = next_relations[range_arr, action_idx]

        return loss, new_state, F.log_softmax(scores, dim=1), action_idx, chosen_relation

    def __call__(self, candidate_relation_sequence, candidate_entity_sequence, current_entities,
                 path_label, query_relation, range_arr, first_step_of_test, T=3, entity_sequence=0):

        self.baseline_inputs = []
        # get the query vector
        query_embedding = self.relation_lookup_table(query_relation)
        state = self.init_memory(self.batch_size, query_embedding.device)

        prev_relation = self.get_dummy_start_label(self.batch_size, query_embedding.device)

        all_loss = []  # list of loss tensors each [B,]
        all_logits = []  # list of actions each [B,]
        action_idx = []

        for t in range(T):
            next_possible_relations = candidate_relation_sequence[t]
            next_possible_entities = candidate_entity_sequence[t]
            current_entities_t = current_entities[t]
            path_label_t = path_label[t]

            loss, state, logits, idx, chosen_relation = self.step(
                next_possible_relations,
                next_possible_entities,
                state,
                prev_relation,
                query_embedding,
                current_entities_t,
                label_action=path_label_t,
                range_arr=range_arr,
                first_step_of_test=first_step_of_test,
            )

            all_loss.append(loss)
            all_logits.append(logits)
            action_idx.append(idx)
            prev_relation = chosen_relation

            # [(B, T), 4D]

        return all_loss, all_logits, action_idx
