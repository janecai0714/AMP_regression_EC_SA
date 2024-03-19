#coding=utf-8
from torch.utils.data import Dataset
import numpy as np

class amp_dataset(Dataset):
    def __init__(self, feature_list, target_list):
        self.features = feature_list
        self.label = target_list
        self.set_length = len(self.label)
    def __getitem__(self, index):
        fea = self.features[index]
        lab = self.label[index]
        return fea, lab
    def __len__(self):
        return len(self.features)