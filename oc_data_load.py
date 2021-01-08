import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from skimage import io, transform

class CIFAR10_Data(Dataset):
    '''Loads a subset of CIFAR10 data specified by a csv_file'''

    def __init__(self, root_dir, csv_file, fold=None, transform=None):
        self.frame = pd.read_csv(csv_file)
        if fold is not None:
            self.frame = self.frame[self.frame['fold'] == fold]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
 
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx,0])
        image = io.imread(img_name)
        # metadata = self.frame.iloc[idx, 1:]
        # sample = {'image': image, 'metadata': metadata}
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample

'''
class CIFAR10_Data(Dataset):
Loads a subset of CIFAR10 data specified by a csv_file
    def __init__(self, root_dir, csv_file, fold=None, transform=None):
        self.frame = pd.read_csv(csv_file)
        if fold is not None:
            self.frame = self.frame[self.frame['fold'] == fold]
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.frame.iloc[idx,0])
        image = io.imread(img_name)
        # metadata = self.frame.iloc[idx, 1:]
        # sample = {'image': image, 'metadata': metadata}
        sample = image

        if self.transform:
            sample = self.transform(sample)

        return sample
'''
