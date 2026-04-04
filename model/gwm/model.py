import torch
import torch.nn as nn

class GWMConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class GWM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Structural Component (Entity/Relation Embeddings)
        self.structural_dim = config.structural_dim
        self.entity_embeddings = nn.Embedding(config.num_entities, self.structural_dim)
        self.relation_embeddings = nn.Embedding(config.num_relations, self.structural_dim)

        self.structural_projection = None
        if self.structural_dim != config.hidden_dim:
            self.structural_projection = nn.Linear(self.structural_dim, config.hidden_dim)
        
        # 2. Path Processing (RNN / GWM Core)
        # Context-Free (No neighborhood nodes), just [Head -> Relation]
        self.lstm = nn.LSTM(
            input_size=config.hidden_dim, 
            hidden_size=config.hidden_dim,
            num_layers=getattr(config, 'num_layers', 1),
            batch_first=True,
            dropout=getattr(config, 'dropout', 0.1) if getattr(config, 'num_layers', 1) > 1 else 0
        )
        
        # 3. Fusion Layer
        text_dim = int(getattr(config, 'text_embedding_dim', config.hidden_dim))
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
        
        # 4. Output Projector
        self.projector = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.cached_entity_text_emb = None
        self.cached_relation_text_emb = None
        self.use_text_cache = False
        self._cached_all_entity_targets = None

    def _load_embedding_tensor(self, source, expected_rows, name):
        if isinstance(source, str):
            loaded = torch.load(source, map_location='cpu')
        elif torch.is_tensor(source):
            loaded = source.detach().cpu()
        else:
            raise TypeError(f"Unsupported {name} cache source: {type(source)}")

        if isinstance(loaded, dict):
            if 'embeddings' in loaded:
                loaded = loaded['embeddings']
            elif 'tensor' in loaded:
                loaded = loaded['tensor']
            else:
                raise ValueError(f"{name} cache dict must contain 'embeddings' or 'tensor'.")

        if not torch.is_tensor(loaded):
            raise TypeError(f"{name} cache must resolve to a torch.Tensor.")

        loaded = loaded.float().contiguous()
        if loaded.dim() != 2:
            raise ValueError(f"{name} cache must be rank-2. Got shape {tuple(loaded.shape)}")
        if loaded.size(0) != expected_rows:
            raise ValueError(
                f"{name} cache row count mismatch. Expected {expected_rows}, got {loaded.size(0)}"
            )
        return loaded

    def load_precomputed_text_embedding_cache(self, entity_source, relation_source, cache_device='cpu'):
        if self.use_text_cache and self.cached_entity_text_emb is not None and self.cached_relation_text_emb is not None:
            return

        cache_device = torch.device(cache_device)

        entity_cache = self._load_embedding_tensor(
            source=entity_source,
            expected_rows=self.entity_embeddings.num_embeddings,
            name='entity',
        ).to(cache_device)

        relation_cache = self._load_embedding_tensor(
            source=relation_source,
            expected_rows=self.relation_embeddings.num_embeddings,
            name='relation',
        ).to(cache_device)

        if entity_cache.size(1) != relation_cache.size(1):
            raise ValueError(
                "Entity and relation text embeddings must share the same embedding dimension. "
                f"Got {entity_cache.size(1)} and {relation_cache.size(1)}"
            )

        expected_text_dim = self.text_projection.in_features
        if entity_cache.size(1) != expected_text_dim:
            raise ValueError(
                "Text embedding dimension mismatch with model config. "
                f"Expected {expected_text_dim}, got {entity_cache.size(1)}"
            )

        self.cached_entity_text_emb = entity_cache
        self.cached_relation_text_emb = relation_cache
        self.use_text_cache = True
        self._cached_all_entity_targets = None

        if cache_device.type == 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Backward-compatible alias for older callers.
    def build_text_embedding_cache(self, entity_source, relation_source, device='cpu', **_kwargs):
        self.load_precomputed_text_embedding_cache(entity_source, relation_source, cache_device=device)

    def _lookup_cached_text(self, ids, kind='entity'):
        cache = self.cached_entity_text_emb if kind == 'entity' else self.cached_relation_text_emb
        if cache is None:
            raise RuntimeError("Text cache is not built. Call load_precomputed_text_embedding_cache first.")

        original_shape = ids.shape
        flat_ids = ids.reshape(-1)
        if flat_ids.device != cache.device:
            flat_ids = flat_ids.to(cache.device)
        selected = cache.index_select(0, flat_ids)
        if selected.device != ids.device:
            selected = selected.to(ids.device)
        return selected.reshape(*original_shape, -1)

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
            raise RuntimeError("Text cache is not built. Call load_precomputed_text_embedding_cache before training/inference.")

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
            raise RuntimeError("Text cache is not built. Call load_precomputed_text_embedding_cache before training/inference.")

        t_emb_text = self._lookup_cached_text(t_batch['id'], kind='entity')
        t_struct = self.entity_embeddings(t_batch['id'])
        t_fused = self._fuse_modalities(t_emb_text, t_struct)
        return torch.nn.functional.normalize(t_fused, p=2, dim=1)

    def compute_loss(self, query_vector, t_fused):
        scores = torch.mm(query_vector, t_fused.t())
        scores /= getattr(self.config, 'temperature', 0.07)
        labels = torch.arange(scores.size(0), device=scores.device)
        return nn.CrossEntropyLoss()(scores, labels), scores

    def _get_all_entity_targets(self, device):
        if self._cached_all_entity_targets is not None and self._cached_all_entity_targets.device == device:
            return self._cached_all_entity_targets

        with torch.no_grad():
            entity_ids = torch.arange(self.entity_embeddings.num_embeddings, device=device, dtype=torch.long)
            targets = self.encode_target({'id': entity_ids})
            self._cached_all_entity_targets = targets
        return self._cached_all_entity_targets

    def predict_latent_jumps(self, current_entities, query_relations, k=3):
        if not self.use_text_cache:
            raise RuntimeError("Text cache is not built. Call load_precomputed_text_embedding_cache first.")

        device = next(self.parameters()).device
        with torch.no_grad():
            curr = torch.as_tensor(current_entities, dtype=torch.long, device=device)
            rel = torch.as_tensor(query_relations, dtype=torch.long, device=device)

            curr = curr.clamp(min=0, max=self.entity_embeddings.num_embeddings - 1)
            rel = rel.clamp(min=0, max=self.relation_embeddings.num_embeddings - 1)

            q = self({'id': curr}, {'id': rel})
            all_entities = self._get_all_entity_targets(device)
            scores = torch.matmul(q, all_entities.t())

            topk = min(int(k), all_entities.size(0))
            _, topk_ids = torch.topk(scores, k=topk, dim=1)

            if topk < int(k):
                pad_col = topk_ids[:, -1:].repeat(1, int(k) - topk)
                topk_ids = torch.cat([topk_ids, pad_col], dim=1)

            return topk_ids.cpu().numpy()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        config_dict = checkpoint.get('config', {})
        required_fields = ['structural_dim', 'hidden_dim', 'num_entities', 'num_relations']
        missing = [f for f in required_fields if f not in config_dict]
        if missing:
            raise ValueError(
                f"Checkpoint config missing required fields: {missing}. "
                "Expected fields: structural_dim, hidden_dim, num_entities, num_relations."
            )

        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
        if state_dict is None:
            raise ValueError("Checkpoint does not contain 'state_dict' or 'model_state_dict'.")

        if 'text_embedding_dim' not in config_dict:
            if 'text_projection.weight' in state_dict:
                config_dict['text_embedding_dim'] = state_dict['text_projection.weight'].shape[1]
            elif 'fusion.weight' in state_dict:
                fusion_in = state_dict['fusion.weight'].shape[1]
                config_dict['text_embedding_dim'] = fusion_in - int(config_dict['structural_dim'])
            else:
                raise ValueError(
                    "Unable to infer text_embedding_dim from checkpoint. "
                    "Please include text_embedding_dim in checkpoint config."
                )

        config = GWMConfig(**config_dict)
        model = cls(config)
        model.load_state_dict(state_dict, strict=False)
        return model