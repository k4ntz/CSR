import torch, os
import numpy as np
from torch.utils.data import Dataset

class DRSSMDataset(Dataset):
    def __init__(self, dataset_path):
        super().__init__()
        dataset = np.load(dataset_path)
        self.data = dataset

    def __len__(self):
        return len(self.data["s1"])

    def __getitem__(self, idx):
        hs1 = torch.from_numpy(self.data["hs1"][idx].astype('float32'))
        hs2 = torch.from_numpy(self.data["hs2"][idx].astype('float32'))
        hs3 = torch.from_numpy(self.data["hs3"][idx].astype('float32'))
        hs4 = torch.from_numpy(self.data["hs4"][idx].astype('float32'))
        s1 = torch.from_numpy(self.data["s1"][idx].astype('float32'))
        s2 = torch.from_numpy(self.data["s2"][idx].astype('float32'))
        s3 = torch.from_numpy(self.data["s3"][idx].astype('float32'))
        s4 = torch.from_numpy(self.data["s4"][idx].astype('float32'))
        sample = {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "hs1": hs1, "hs2": hs2, "hs3": hs3, "hs4": hs4, }
        return sample