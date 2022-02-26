


#%% Import Libraries ===============================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.autograd import Variable
from datetime import datetime
import os

from model import NN_MNIST

# Colab~~~~~~~~~~~~~~~~~ 


# Set path for model and data locations
path_data = './'
path_model = './'




# Define function for calculating predictions on test data

def calc_test_accy(model, test_loader):
    model.eval()   # Set model into evaluation mode
    correct = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(
                device)  # Move data to GPU
            output = model(data)   # Calculate Output
            pred = output.max(1, keepdim=True)[1]  # Calculate Predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

        return (100.*correct/len(test_loader.dataset))


# Define function for calculating predictions on training data
# n_batches indicates how many training observations to use for training accuracy calculation
# This reduces computation time, instead of using the entire training set.
def calc_train_accy(model, dataloader, n_batches, batch_size):
    model.eval()
    correct = 0
    with torch.no_grad():
        data_iterator = iter(dataloader)
        for i in range(n_batches):  # iterate for the specified number of batches
            try:
                data, target = next(data_iterator)
            except StopIteration:
                data_iterator = iter(dataloader)
                data, target = next(data_iterator)

            data, target = data.to(device), target.to(
                device)  # Move data to GPU
            output = model(data)   # Calculate Output
            pred = output.max(1, keepdim=True)[1]  # Calculate Predictions
            correct += pred.eq(target.view_as(pred)).sum().item()

    return (100.*correct/(n_batches * batch_size))


#%%  Model Parameters ===================================================
load_checkpoint = False
num_epochs = 20
batch_size = 32
optimizer_params = {
    'lr': 0.01,
    'weight_decay': 0,
    'momentum': 0
}
scheduler_params = {
    'step_size': 8,
    'gamma': 0.1
}
model_params = {
    'input_units':28*28,
    'num_classes':10,
    'hidden_units':500,
    'n_layers':3
}


#%%  Load Data =================================================


# Define Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.1307], std=[0.3081])
    ]),

    'test': transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
}


# Training Data
train_set = datasets.MNIST(
    root=path_data, train=True, download=True, transform=data_transforms['train'])
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# Test Data
test_set = datasets.MNIST(
    root=path_data, train=False, download=True, transform=data_transforms['test'])
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=0)


#%%  Run Model ==================================================

# Initialize Model
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")  # Use GPU if available
model = NN_MNIST(**model_params).to(device)

# Initialize Optimizer
optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
criterion = nn.CrossEntropyLoss()
optimizer.zero_grad()

# Initialize Other Trackers
StartTime = datetime.now()  # track training time
# numpy array to track training/test accuracy by epoch
save_accy = np.zeros((num_epochs + 1, 2))
np.random.seed(1)
torch.manual_seed(1)
# Epoch to start calculations (should be zero unless loading previous model checkpoint)
init_epoch = 0
n_batches = np.int(np.floor((len(train_set)*1.)/batch_size))

#~~~~~~~~~~~~~~~~~~~~~~
# If we are starting from a loaded checkpoint, load previous model paramters
if load_checkpoint:
    checkpoint = torch.load(os.path.join(path_model, 'checkpoint.pt'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    init_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    model.to(device)
#~~~~~~~~~~~~~~~~~~~~~~~

for epoch in range(init_epoch, num_epochs):

    epoch_time_start = datetime.now()
    model.train()
    for batch_ID, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)  # Move training data to GPU
        pred = model(data)      # Calculate Predictions
        loss = criterion(pred, target)  # Calculate Loss
        loss.backward()  # Calculate Gradients

        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    # Calculate Train/Test Accuracy
    test_accy = calc_test_accy(model, test_loader)
    train_accy = calc_train_accy(model, train_loader, np.minimum(25, n_batches), batch_size)
    # Save training / test accuracy
    save_accy[epoch, :] = [train_accy, test_accy]
    epoch_time = datetime.now() - epoch_time_start
    print("Epoch: %d || Loss: %f || Train_Accy: %f || Test_Accy: %f || Time: %s" %
          (epoch, loss, train_accy, test_accy, str(epoch_time)))

    # Save Checkpoint
    if epoch % 5 == 0:
        checkpoint = {'epoch': epoch,
                      'model': NN_MNIST(**model_params),
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(),
                      'scheduler': scheduler.state_dict()
                      }

        # Write Checkpoint to disk
        torch.save(checkpoint, os.path.join(path_model, 'checkpoint.pt'))
        np.savetxt(os.path.join(path_model, 'model_accuracy.csv'),
                   save_accy, delimiter=",")  # Write training/test accuracy to disk

#%% Evaluate Final Model ==================================================

test_accy = calc_test_accy(model, test_loader)
train_accy = calc_test_accy(model, train_loader)
save_accy[-1, :] = [train_accy, test_accy]

total_time = datetime.now() - StartTime  # Calculate total training / test time
print("Train Accuracy: %f; Test Accuracy: %f; Total Time: %s; Epochs: %d" %
      (train_accy, test_accy, total_time, epoch))
# Save final model to disk
torch.save(model, os.path.join(path_model, 'MLP_baseline.pt'))
np.savetxt(os.path.join(path_model, 'model_accuracy.csv'), save_accy,
           delimiter=",")  # Write training/test accuracy to disk
