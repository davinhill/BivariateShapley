import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist

import time
import os
import sys
import io
from tqdm import tqdm

from model_RNNClassifier import RNN
from load_data import load_data
from utils import *




opt = 'adam'
num_epochs = 15
LR = 0.0001
batch_size = 128
vocab_size = 10000
hidden_units = 500
max_length = 400
word_feat = 500
freeze_embed = False
use_glove = False

path = os.path.dirname(os.path.realpath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RNN(vocab_size, hidden_units, dropout=0, num_layers = 1, word_feat = word_feat, freeze_embed = freeze_embed, use_glove = use_glove)
model.to(device)

if(opt == 'adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt == 'sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

train_loader, val_loader, _ = load_data(batch_size = batch_size, vocab_size = vocab_size, max_length = max_length, path = os.path.join(path, './preprocessed_data'))
criterion = nn.BCEWithLogitsLoss()
optimizer.zero_grad()

train_loss = []
train_accy = []
val_accy = []
num_batches = len(train_loader)

for epoch in range(num_epochs):

    model.train()

    # trackers
    epoch_accy = 0.0
    epoch_loss = 0.0
    epoch_counter = 0
    time1 = time.time()

    for idx, batch in tqdm(enumerate(train_loader), total = num_batches):
        data, target = batch[0].to(device), batch[1].to(device)
        data_lens = tensor2cuda(batch[2])

        pred = model(data, data_lens)
        loss = criterion(pred, target)
        loss.backward()

        optimizer.step()   # update weights
        optimizer.zero_grad()

        if epoch == 10:
            model.embedding0.weight.requires_grad = True
        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_accy += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_accy /= (num_batches*batch_size)
    epoch_loss /= num_batches

    train_loss.append(epoch_loss)
    train_accy.append(epoch_accy)

    # calculate validation accuracy
    val_accy = calc_test_accy_rnn(model, val_loader)

    time2 = time.time()
    time_elapsed = time2 - time1
    

    print("Epoch: %.2f  || Train_Loss: %.4f  ||  Train_Accy: %.4f  ||  Val_Accy: %f  ||  Time: %f" % (
        epoch, epoch_loss, epoch_accy*100, val_accy*100, float(time.time()-time1)))

torch.save(model, 'RNN_model.pt')
data = [train_loss, train_accy, val_accy]
data = np.asarray(data)
np.save('data.npy', data)
