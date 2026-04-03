import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class GWMConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 1. Text Encoder (always frozen, used only for one-time cache building)
        self.tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model)
        self.text_encoder = AutoModel.from_pretrained(config.pretrained_model)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.text_encoder.eval()
                
        # 2. Structural Component (Entity/Relation Embeddings)
        self.structural_dim = config.structural_dim
        self.entity_embeddings = nn.Embedding(config.num_entities, self.structural_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, self.structural_dim)

        self.structural_projection = None
        if self.structural_dim != config.hidden_dim:
            self.structural_projection = nn.Linear(self.structural_dim, config.hidden_dim)
        
        # 3. Path Processing (RNN / GWM Core)
        # Context-Free (No neighborhood nodes), just [Head -> Relation]
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim, 
            hidden_size=config.hidden_dim,
            num_layers=getattr(config, 'num_layers', 1),
            batch_first=True,
            dropout=getattr(config, 'dropout', 0.1) if getattr(config, 'num_layers', 1) > 1 else 0
        )
        
        # 4. Fusion Layer
        text_dim = self.text_encoder.config.hidden_size
        self.text_projection = nn.Linear(text_dim, config.hidden_dim)
        self.fusion_mode = getattr(config, 'fusion_mode', 'concat')

        self.fusion = nn.Linear(text_dim + self.structural_dim, config.hidden_dim)

        if self.fusion_mode == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, 1),
                nn.Sigmoid()
            )

        self.reset_alpha_stats()
        
        # 5. Output Projector
        self.projector = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.cached_entity_text_emb = None
        self.cached_relation_text_emb = None
        self.use_text_cache = False
        
    def _encode_text(self, input_ids, attention_mask):
        if self.text_encoder is None:
            raise RuntimeError("Text encoder is released. Build cache before using ID lookups.")
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :] # [CLS] token

    def build_text_embedding_cache(self, entity_text_map, relation_text_map, device, batch_size=128, max_entity_length=512, max_relation_length=128):
        if self.use_text_cache and self.cached_entity_text_emb is not None and self.cached_relation_text_emb is not None:
            return

        if self.text_encoder is None:
            raise RuntimeError("Text encoder is not available and cache is missing.")

        self.text_encoder.to(device)
        self.text_encoder.eval()

        def _encode_text_list(text_list, max_length):
            all_emb = []
            with torch.no_grad():
                for start in range(0, len(text_list), batch_size):
                    chunk = text_list[start:start + batch_size]
                    enc = self.tokenizer(chunk, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
                    enc = {k: v.to(device) for k, v in enc.items()}
                    emb = self._encode_text(enc['input_ids'], enc['attention_mask'])
                    all_emb.append(emb)
            return torch.cat(all_emb, dim=0).contiguous()

        num_entities = self.entity_embeddings.num_embeddings
        num_relations = self.relation_embeddings.num_embeddings

        entity_texts = [entity_text_map.get(str(i), f"Entity {i}") for i in range(num_entities)]
        relation_texts = [relation_text_map.get(str(i), f"Relation {i}") for i in range(num_relations)]

        self.cached_entity_text_emb = _encode_text_list(entity_texts, max_entity_length)
        self.cached_relation_text_emb = _encode_text_list(relation_texts, max_relation_length)
        self.use_text_cache = True

        if getattr(self.config, 'drop_text_encoder_after_cache', True):
            self.text_encoder = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _lookup_cached_text(self, ids, kind='entity'):
        cache = self.cached_entity_text_emb if kind == 'entity' else self.cached_relation_text_emb
        if cache is None:
            raise RuntimeError("Text cache is not built. Call build_text_embedding_cache first.")

        original_shape = ids.shape
        flat_ids = ids.view(-1)
        selected = cache.index_select(0, flat_ids)
        return selected.view(*original_shape, -1)

    def _project_structural(self, struct_emb):
        if self.structural_projection is not None:
            return self.structural_projection(struct_emb)
        return struct_emb

    def reset_alpha_stats(self):
        self._alpha_sum = 0.0
        self._alpha_count = 0

    def get_alpha_mean(self, reset=False):
        if self.fusion_mode != 'gated' or self._alpha_count == 0:
            alpha_mean = None
        else:
            alpha_mean = self._alpha_sum / self._alpha_count
        if reset:
            self.reset_alpha_stats()
        return alpha_mean

    def _fuse_modalities(self, text_emb, struct_emb):
        if self.fusion_mode == 'gated':
            text_proj = self.text_projection(text_emb)
            struct_proj = self._project_structural(struct_emb)
            gate_input = torch.cat([text_proj, struct_proj], dim=-1)
            alpha = self.gate(gate_input)
            alpha_detached = alpha.detach()
            self._alpha_sum += alpha_detached.sum().item()
            self._alpha_count += alpha_detached.numel()
            return alpha * text_proj + (1.0 - alpha) * struct_proj

        return self.fusion(torch.cat([text_emb, struct_emb], dim=-1))
        
    def forward(self, h_batch, r_batch):
        """
        Forward pass without neighborhood context aggregation predicting tail.
        h_batch: dict {'id'}
        r_batch: dict {'id'}
        """
        if not self.use_text_cache:
            raise RuntimeError("Text cache is not built. Call build_text_embedding_cache before training/inference.")

        h_emb_text = self._lookup_cached_text(h_batch['id'], kind='entity')
        r_emb_text = self._lookup_cached_text(r_batch['id'], kind='relation')
        
        h_struct = self.entity_embeddings(h_batch['id']) # (B, H)
        r_struct = self.relation_embeddings(r_batch['id']) # (B, H)
        
        h_fused = self._fuse_modalities(h_emb_text, h_struct) # (B, H)
        r_fused = self._fuse_modalities(r_emb_text, r_struct) # (B, H)

        # Context-Free LSTM Sequence: Just [Head, Relation] -> Predict Tail
        lstm_input = torch.stack([h_fused, r_fused], dim=1) # (B, 2, H)
        
        lstm_out, _ = self.lstm(lstm_input)
        query_vector = lstm_out[:, -1, :] # Last hidden state (B, H)
        
        query_vector = self.projector(query_vector)
        query_vector = torch.nn.functional.normalize(query_vector, p=2, dim=1)
        
        return query_vector

    def encode_target(self, t_batch):
        if not self.use_text_cache:
            raise RuntimeError("Text cache is not built. Call build_text_embedding_cache before training/inference.")

        t_emb_text = self._lookup_cached_text(t_batch['id'], kind='entity')
        t_struct = self.entity_embeddings(t_batch['id'])
        t_fused = self._fuse_modalities(t_emb_text, t_struct)
        return torch.nn.functional.normalize(t_fused, p=2, dim=1)

    def compute_loss(self, query_vector, t_fused):
        scores = torch.mm(query_vector, t_fused.t())
        scores /= getattr(self.config, 'temperature', 0.07)
        labels = torch.arange(scores.size(0), device=scores.device)
        return nn.CrossEntropyLoss()(scores, labels), scores

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config_dict = checkpoint.get('config', {})
        required_fields = ['pretrained_model', 'structural_dim', 'hidden_dim', 'num_entities', 'num_relations']
        missing = [f for f in required_fields if f not in config_dict]
        if missing:
            raise ValueError(
                f"Checkpoint config missing required fields: {missing}. "
                "Expected fields: pretrained_model, structural_dim, hidden_dim, num_entities, num_relations."
            )

        config = GWMConfig(**config_dict)
        model = cls(config)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
        if state_dict is None:
            raise ValueError("Checkpoint does not contain 'state_dict' or 'model_state_dict'.")
        model.load_state_dict(state_dict, strict=False)
        return model