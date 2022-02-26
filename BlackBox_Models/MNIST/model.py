
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class NN_MNIST(nn.Module):
    # Define ResNet

    def __init__(self, input_units, num_classes, hidden_units, n_layers):
        super(NN_MNIST, self).__init__()

        self.input_units = input_units
        self.num_classes = num_classes
        self.hidden_units = hidden_units
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 200, kernel_size = 6, stride = 2, padding = 0) # 12 x 12 x 300
        self.conv2 = nn.Conv2d(in_channels = 200, out_channels = 200, kernel_size = 6, stride = 2, padding = 0) # 4 x 4 x 500

        self.bn1 = nn.BatchNorm2d(200)
        self.fc1 = nn.Linear(3200, self.num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, start_dim = 1)
        x = self.fc1(x)

        return x
