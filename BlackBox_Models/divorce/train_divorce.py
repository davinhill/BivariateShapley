


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

from model import NN, divorce

# Set path for model and data locations
path_data = '../Data/divorce.h5'
path_model = os.path.dirname(os.path.realpath(__file__))




#%% Define Classes ===============================================


# Define function for calculating predictions on test data

def calc_test_accy(model, test_loader):
    model.eval()   # Set model into evaluation mode
    epoch_accy = 0
    counter = 0
    with torch.no_grad():
        for (data, target) in test_loader:
            data, target = data.to(device), target.to(
                device)  # Move data to GPU
            pred = model(data)   # Calculate Output
            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_accy += acc
            counter += target.shape[0]
        epoch_accy /= counter
        return epoch_accy 



#%%  Model Parameters ===================================================
load_checkpoint = False
num_epochs = 30
batch_size = 85
optimizer_params = {
    'lr':1e-1,
    'weight_decay': 0,
    'momentum': 0
}
scheduler_params = {
    'step_size': 20,
    'gamma': 0.1
}
model_params = {
    'input_units':54,
    'hidden_units':50,
    'num_classes':1
}


#%%  Load Data =================================================


train_set = divorce(data_path = path_data, train = True)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=len(train_set), shuffle=True, num_workers=0)

test_set = divorce(data_path = path_data, train = False)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=len(test_set), shuffle=False, num_workers=0)


#%%  Run Model ==================================================

# Initialize Model
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")  # Use GPU if available
model = NN(**model_params).to(device)

# Initialize Optimizer
optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
criterion = nn.BCEWithLogitsLoss()
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
        data, target = data.to(device), target.to(device).double()  # Move training data to GPU

        pred = model(data)      # Calculate Predictions
        loss = criterion(pred, target)  # Calculate Loss
        loss.backward()  # Calculate Gradients

        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    # Calculate Train/Test Accuracy
    test_accy = calc_test_accy(model, test_loader)
    train_accy = calc_test_accy(model, train_loader)
    # Save training / test accuracy
    save_accy[epoch, :] = [train_accy, test_accy]
    epoch_time = datetime.now() - epoch_time_start
    print("Epoch: %d || Loss: %f || Train_Accy: %f || Test_Accy: %f || Time: %s" %
          (epoch, loss, train_accy, test_accy, str(epoch_time)))

    # Save Checkpoint
    if epoch % 5 == 0:
        checkpoint = {'epoch': epoch,
                      'model': NN(**model_params),
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
torch.save(model, os.path.join(path_model, 'divorce_model.pt'))
