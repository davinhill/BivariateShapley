import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import os
import io


class IMDB(Dataset):

    def __init__(self, vocab_size = 8000, max_length = 500, train = True, labels_only = True, path= './preprocessed_data', padding = False):
        """
        args:
        """

        self.max_length = max_length
        self.padding = padding

        if train:
            # there are 75k sentences in training set, but only 25k are labeled.
            x = []
            with io.open(os.path.join(path, 'imdb_train.txt'), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(' ')
                line = np.asarray(line, dtype=np.int)

                line[line >= vocab_size] = 1  # words not in vocab set to <UNK>

                line = line[:max_length]  # limit to max word length

                x.append(line)

            if labels_only:
                x = x[0:25000]
                y = np.zeros((25000,))
                y[0:12500] = 1
            else:
                y = np.zeros((75000, 1))

        else:
            x = []
            with io.open(os.path.join(path, 'imdb_test.txt'), 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split(' ')
                line = np.asarray(line, dtype=np.int)

                line[line >= vocab_size] = 1  # words not in vocab set to <UNK>

                line = line[:max_length]  # limit to max word length

                x.append(line)
            y = np.zeros((25000,))
            y[0:12500] = 1

        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = torch.tensor(self.data[idx])
        y = torch.tensor(self.label[idx])
        if self.padding:
            if len(x) < self.max_length:
                x = torch.cat((x, torch.zeros(self.max_length - len(x), dtype = torch.int)))
        return x, y


def pad_collate(batch):
    '''
    collates sentences of different lengths
    adapted from https://suzyahyah.github.io/pytorch/2019/07/01/DataLoader-Pad-Pack-Sequence.html

    '''
    # pad sequences and collate
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy = torch.tensor(yy)
    
    # Pre-sort sequences by length
    x_lens = torch.from_numpy(np.array(x_lens))
    sorted_lengths, sorted_idx = torch.sort(x_lens, descending=True)
    xx_pad = xx_pad[sorted_idx,...]
    yy = yy[sorted_idx]
    x_lens = sorted_lengths

    return xx_pad, yy, x_lens



def load_data(vocab_size = 8000, max_length = 500,path= './preprocessed_data', batch_size = 128, labels_only = True, val_split = 5000, padding = False):
    train_set = IMDB(vocab_size, train = True, max_length = max_length, path = path, labels_only = labels_only, padding = padding)
    test_set = IMDB(vocab_size, train = False, max_length = max_length, path = path, labels_only = labels_only, padding = padding)

    if labels_only:
        train_size = 25000
    else:
        train_size = 75000

    if padding:
        collate_fn = None
    else:
        collate_fn = pad_collate
    if val_split >0:
        train_set, val_set = random_split(train_set, [train_size - val_split,val_split])
        val_loader = DataLoader(val_set, batch_size = batch_size, shuffle = True, num_workers = 0, collate_fn=collate_fn)
    else:
        val_loader = None

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 0, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = True, num_workers = 0, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

