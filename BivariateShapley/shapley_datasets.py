import torch
import numpy as np

import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
import os

class Syn_1(Dataset):

    def __init__(self, n_samples, n_features):
        self.n_samples = n_samples
        
        mean = np.zeros(n_features)
        cov = np.eye(n_features)
        self.data = np.random.multivariate_normal(mean, cov, n_samples)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :], np.zeros((1, 5))

class Syn_2(Dataset):
    'gaussian features, plus redundant features'

    def __init__(self, n_samples, n_features):
        self.n_samples = n_samples
        
        mean = np.zeros(n_features)
        cov = np.eye(n_features)
        self.data = np.random.multivariate_normal(mean, cov, n_samples)
        self.data = np.concatenate((self.data, np.sum(self.data, axis = 1, keepdims=True)), axis = 1)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :], np.zeros((1, 5))

class glove(Dataset):

    def __init__(self, n_samples, n_features, prob_R = 0.5):
        self.n_samples = n_samples
        
        p = np.ones(n_features) * prob_R
        sample = np.expand_dims(np.random.binomial(1, p), 0)

        for i in range(n_samples):
            sample = np.concatenate((sample, np.expand_dims(np.random.binomial(1, p), 0)), axis = 0)
        
        self.data = sample

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :], np.zeros((1, 5))

class divorce(Dataset):

    def __init__(self, data_path = './Data/divorce.h5', train = True):
        import h5py

        hf = h5py.File(data_path, 'r')

        if train:
            self.data = torch.tensor(hf.get('data_train'), dtype = torch.float32)
            self.labels = torch.tensor(hf.get('label_train'))
            #self.ind = torch.tensor(hf.get('idx_train'))  # selected indices of original dataset for reference
        else:

            self.data = torch.tensor(hf.get('data_test'), dtype = torch.float32)
            self.labels = torch.tensor(hf.get('label_test'))
            #self.ind = torch.tensor(hf.get('idx_test'))  # selected indices of original dataset for reference

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :], self.labels[idx]


class COPD(Dataset):

    def __init__(self, data_path = './Data/COPD.h5', train = True):
        import h5py

        hf = h5py.File(data_path, 'r')

        if train:
            self.data = torch.tensor(hf.get('data_train'), dtype = torch.float32)
            self.labels = torch.tensor(hf.get('label_train'))
        else:

            self.data = torch.tensor(hf.get('data_test'), dtype = torch.float32)
            self.labels = torch.tensor(hf.get('label_test'))

        self.gene_names = np.array(hf.get('gene_name'))  # selected indices of original dataset for reference
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :], self.labels[idx]


class drug(Dataset):

    def __init__(self, data_path = './Data/drug.h5', train = True):
        import h5py

        hf = h5py.File(data_path, 'r')

        if train:
            self.data =np.array(hf.get('data_train'))
            self.labels = np.array(hf.get('label_train'))
        else:

            self.data =np.array(hf.get('data_test'))
            self.labels = np.array(hf.get('label_test'))

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :], self.labels[idx]


class census(Dataset):

    def __init__(self, data_path = './Data/', train = True):

        if train:
            self.data = torch.from_numpy(pd.read_pickle(os.path.join(data_path, './census_x_train.pkl')).values)
            self.labels = torch.from_numpy(np.loadtxt(os.path.join(data_path, './census_y_train.csv')))
        else:
            self.data = torch.from_numpy(pd.read_pickle(os.path.join(data_path, './census_x_test.pkl')).values)
            self.labels = torch.from_numpy(np.loadtxt(os.path.join(data_path, './census_y_test.csv')))


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :], self.labels[idx]
