import torch
from torch.utils.data import Dataset
import os

class GWMDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        """
        Dataset for GWM.
        """
        self.data_dir = data_dir
        self.split = split
        
        # Load triples
        triples_path = os.path.join(data_dir, f'{split}_triples.pt')       
        self.triples = torch.load(triples_path)

    def __len__(self):
        return len(self.triples)
        
    def __getitem__(self, idx):
        h, r, t = self.triples[idx]

        return {
            'h_id': h.long(),
            'r_id': r.long(),
            't_id': t.long(),
        }

class CollateFN:
    """
    ID-only collator. Text encoding is handled offline by cache builder.
    """
    def __init__(self):
        pass
        
    def __call__(self, batch):
        h_ids = torch.stack([b['h_id'] for b in batch])
        r_ids = torch.stack([b['r_id'] for b in batch])
        t_ids = torch.stack([b['t_id'] for b in batch])
        
        return {
            'h_batch': {'id': h_ids},
            'r_batch': {'id': r_ids},
            't_batch': {'id': t_ids},
        }
