import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return {
            "src": self.src_data[idx],
            "tgt": self.tgt_data[idx]
        }
