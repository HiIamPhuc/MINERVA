import torch
import torch.nn as nn
import contextlib
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
        self._cached_all_entity_targets = None
        
    def _encode_text(self, input_ids, attention_mask):
        if self.text_encoder is None:
            raise RuntimeError("Text encoder is released. Build cache before using ID lookups.")
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :] # [CLS] token

    def build_text_embedding_cache(
        self,
        entity_text_map,
        relation_text_map,
        device,
        batch_size=128,
        max_entity_length=512,
        max_relation_length=128,
        encode_device=None,
        cache_device=None,
        autocast_text_encoder=True,
    ):
        if self.use_text_cache and self.cached_entity_text_emb is not None and self.cached_relation_text_emb is not None:
            return

        if self.text_encoder is None:
            raise RuntimeError("Text encoder is not available and cache is missing.")

        runtime_device = torch.device(device)
        encode_device = torch.device(encode_device) if encode_device is not None else runtime_device
        cache_device = torch.device(cache_device) if cache_device is not None else runtime_device

        self.text_encoder.to(encode_device)
        if runtime_device.type == 'cuda' and encode_device.type != 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.text_encoder.eval()

        def _encode_text_list(text_list, max_length):
            all_emb = []
            start = 0
            effective_batch_size = max(1, int(batch_size))

            with torch.inference_mode():
                while start < len(text_list):
                    end = min(start + effective_batch_size, len(text_list))
                    chunk = text_list[start:end]

                    try:
                        enc = self.tokenizer(
                            chunk,
                            padding=True,
                            truncation=True,
                            return_tensors='pt',
                            max_length=max_length,
                        )
                        enc = {k: v.to(encode_device) for k, v in enc.items()}

                        use_autocast = encode_device.type == 'cuda' and autocast_text_encoder
                        autocast_ctx = (
                            torch.autocast(device_type='cuda', dtype=torch.float16)
                            if use_autocast else contextlib.nullcontext()
                        )

                        with autocast_ctx:
                            emb = self._encode_text(enc['input_ids'], enc['attention_mask'])

                        all_emb.append(emb.to(cache_device, non_blocking=True).contiguous())
                        start = end
                    except torch.OutOfMemoryError:
                        if encode_device.type != 'cuda' or effective_batch_size == 1:
                            raise
                        torch.cuda.empty_cache()
                        effective_batch_size = max(1, effective_batch_size // 2)

            return torch.cat(all_emb, dim=0).contiguous()

        num_entities = self.entity_embeddings.num_embeddings
        num_relations = self.relation_embeddings.num_embeddings

        entity_texts = [entity_text_map.get(str(i), f"Entity {i}") for i in range(num_entities)]
        relation_texts = [relation_text_map.get(str(i), f"Relation {i}") for i in range(num_relations)]

        self.cached_entity_text_emb = _encode_text_list(entity_texts, max_entity_length)
        self.cached_relation_text_emb = _encode_text_list(relation_texts, max_relation_length)
        self.use_text_cache = True
        self._cached_all_entity_targets = None

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
        if flat_ids.device != cache.device:
            flat_ids = flat_ids.to(cache.device, non_blocking=True)

        selected = cache.index_select(0, flat_ids)

        if selected.device != ids.device:
            selected = selected.to(ids.device, non_blocking=True)

        target_dtype = self.text_projection.weight.dtype
        if selected.dtype != target_dtype:
            selected = selected.to(dtype=target_dtype)

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
            raise RuntimeError("Text cache is not built. Call build_text_embedding_cache first.")

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