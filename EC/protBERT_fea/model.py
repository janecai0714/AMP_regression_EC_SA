import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn import Conv2d
from collections import OrderedDict


class regNet(nn.Module):
    def __init__(self):
        super(regNet, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1*1024, 512),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits.squeeze()

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