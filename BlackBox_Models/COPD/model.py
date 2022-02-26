import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class NN(nn.Module):
    # Define ResNet

    def __init__(self, input_units, hidden_units, num_classes, num_layers = 1, dropout_p = 0):
        super(NN, self).__init__()

        self.input_units = input_units
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.fc_in = nn.Linear(self.input_units, self.hidden_units) 
        self.bn1 = nn.BatchNorm1d(self.hidden_units)
        self.drop1 = nn.Dropout(p=dropout_p)

        fc_layers = []
        for i in range(num_layers):
            fc_layers.append(nn.Linear(self.hidden_units, self.hidden_units))
            fc_layers.append(nn.ReLU())
            #fc_layers.append(nn.BatchNorm1d(self.hidden_units))
            if dropout_p >0: fc_layers.append(nn.Dropout(p=dropout_p))
        self.fc_layers = nn.Sequential(*fc_layers)

        self.bn2 = nn.BatchNorm1d(self.hidden_units)
        self.fc_out = nn.Linear(self.hidden_units, self.num_classes)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        #x = self.bn1(x)
        x = self.fc_layers(x)
        #x = self.bn2(x)
        #x = self.drop1(x)
        x = self.fc_out(x)

        return x.reshape(-1)



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
