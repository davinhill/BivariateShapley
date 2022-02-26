
#%% Import Libraries ===============================================

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.autograd import Variable
from datetime import datetime
import os




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
# num_batches indicates how many training observations to use for training accuracy calculation
# This reduces computation time, instead of using the entire training set.
def calc_train_accy(model, dataloader, num_batches, batch_size):
    model.eval()
    correct = 0
    with torch.no_grad():
        data_iterator = iter(dataloader)
        for i in range(num_batches):  # iterate for the specified number of batches
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

    return (100.*correct/(num_batches * batch_size))


#%%  Model Parameters ===================================================
load_checkpoint = False
num_epochs = 80
batch_size = 128
model_params = {
    'num_classes': 10
}
optimizer_params = {
    'lr': 0.001,
    'weight_decay': 0
}
scheduler_params = {
    'step_size': 60,
    'gamma': 0.1
}

# Set path for model and data locations
path_data = "./"
path_model = "./"


#%%  Load Data =================================================


# Define Data Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue = 0.05),   # Change brightness, contrast, and saturation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),

    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
}


# Training Data
train_set = datasets.CIFAR10(
    root=path_data, train=True, download=True, transform=data_transforms['train'])
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=0)

# Test Data
test_set = datasets.CIFAR10(
    root=path_data, train=False, download=True, transform=data_transforms['test'])
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=0)


#%%  Run Model ==================================================

# Initialize Model
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")  # Use GPU if available
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)

# Initialize Optimizer
optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
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

# If we are starting from a loaded checkpoint, load previous model paramters
if load_checkpoint:
    checkpoint = torch.load(os.path.join(path_model, 'checkpoint.pt'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    init_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    model.to(device)


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
    train_accy = calc_train_accy(model, train_loader, 100, batch_size)
    # Save training / test accuracy
    save_accy[epoch, :] = [train_accy, test_accy]
    epoch_time = datetime.now() - epoch_time_start
    print("Epoch: %d; Loss: %f; Train_Accy: %f; Test_Accy: %f; Time: %s" %
          (epoch, loss, train_accy, test_accy, str(epoch_time)))

#%% Evaluate Final Model ==================================================

test_accy = calc_test_accy(model, test_loader)
train_accy = calc_test_accy(model, train_loader)
save_accy[-1, :] = [train_accy, test_accy]

import pdb; pdb.set_trace()
total_time = datetime.now() - StartTime  # Calculate total training / test time
print("Train Accuracy: %f; Test Accuracy: %f; Total Time: %s; Epochs: %d" %
      (train_accy, test_accy, total_time, epoch))
# Save final model to disk
torch.save(model, os.path.join(path_model, 'MLP_baseline_CIFAR10.pt'))
np.savetxt(os.path.join(path_model, 'model_accuracy.csv'), save_accy,
           delimiter=",")  # Write training/test accuracy to disk
