import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F


class NN(nn.Module):
    # Define ResNet

    def __init__(self, input_units, hidden_units, num_classes):
        super(NN, self).__init__()

        self.input_units = input_units
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.fc1 = nn.Linear(self.input_units, self.hidden_units) 
        self.fc2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.fc3 = nn.Linear(self.hidden_units, self.num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.reshape(-1)



class divorce(Dataset):

    def __init__(self, data_path = './Data/divorce.h5', train = True):
        import h5py

        hf = h5py.File(data_path, 'r')

        if train:
            self.data = torch.tensor(hf.get('data_train'), dtype = torch.float32)
            self.labels = torch.tensor(hf.get('label_train'))
            self.ind = torch.tensor(hf.get('idx_train'))  # selected indices of original dataset for reference
        else:

            self.data = torch.tensor(hf.get('data_test'), dtype = torch.float32)
            self.labels = torch.tensor(hf.get('label_test'))
            self.ind = torch.tensor(hf.get('idx_test'))  # selected indices of original dataset for reference

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx, :], self.labels[idx]