import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import *



class RNN(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units, word_feat = 300, num_layers = 1, dropout = 0, freeze_embed = False, use_glove = False):
        super(RNN, self).__init__()
        self.num_layers = num_layers

        # Embedding Layers
        if not use_glove: 
            # if using self-created word embedding
            self.embedding0 = nn.Embedding(vocab_size, word_feat)
            self.embedding1 = (nn.Linear(word_feat, no_of_hidden_units))

            if freeze_embed:
                print('Freezing embedding weights...')
                self.embedding0.weight.requires_grad = False  # freeze parameters

        else:
            # if using pretrained glove features:
            self.embedding0 = nn.Embedding(vocab_size, 300)
            self.embedding1 = (nn.Linear(300, no_of_hidden_units))

            print('Loading glove features...')
            glove_feat = np.load('preprocessed_data/glove_embeddings.npy')  # load glove features
            glove_feat = glove_feat[:vocab_size, :]  # truncate to vocab_size

            self.embedding0.weight.data.copy_(torch.from_numpy(glove_feat))  # load glove features into embedding layer

            if freeze_embed:
                print('Freezing embedding weights...')
                self.embedding0.weight.requires_grad = False  # freeze parameters    


        self.rnn = nn.GRU(no_of_hidden_units, no_of_hidden_units, num_layers = num_layers, dropout = dropout, batch_first=True)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

    def forward(self, x, x_lens):
        batch_size = x.shape[0]

        embed = self.embedding0(x)  # n x t x features
        embed = self.embedding1(embed) 

        packed_x = pack_padded_sequence(embed, x_lens.data.tolist(), batch_first=True, enforce_sorted=False)
        _, packed_h = self.rnn(packed_x)
        h = packed_h[self.num_layers-1,...]

        h = self.fc_output(h)

        return h[:, 0]  
