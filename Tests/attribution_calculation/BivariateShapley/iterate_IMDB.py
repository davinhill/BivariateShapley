###########
# file imports / path issues
import os
import sys
from pathlib import Path

path = Path(os.path.abspath(__file__)).parents[3]
os.chdir(path)
sys.path.append('./BivariateShapley')

from shapley_sampling import Shapley_Sampling
from shapley_datasets import *
from utils_shapley import *
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from tqdm import tqdm
import time

############################################
# Define Test Parameters
############################################

parser = ArgumentParser(description='get phi plus matrices')

parser.add_argument('--m', type=int, default = 1000,
                    help='number of MC samples')

parser.add_argument('--verbose', action='store_true', default=False,
                    help='boolean, use tqdm')

parser.add_argument('--dataset_min_index', type = int,default=0,
                    help='iterate over dataset starting from min_index')

parser.add_argument('--dataset_samples', type = int,default=50,
                    help='number of samples, starting from min_index')

parser.add_argument('--baseline', type = str,default='zero')
parser.add_argument('--model_type', type = str,default='RNN')
parser.add_argument('--method_name', type = str,default='BivariateShapley')
args = parser.parse_args()

############################################
# Define Tests
############################################

# Parameters
m = args.m
min_index = args.dataset_min_index
max_index = min_index + args.dataset_samples
baseline = args.baseline

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

method_name = args.method_name
save_path = './Files/results_attribution/IMDB_%s' % (method_name)
data_path = './Files/Data/IMDB/preprocessed_data'
sys.path.append('./BlackBox_Models/IMDB')
make_dir(save_path)


vocab_size = 10000

from shapley_explainers import IMDB_Explainer_Binary
from load_data import IMDB, pad_collate

if args.model_type == 'RNN':
    model_path = './Files/trained_bb_models/RNN_model.pt'
    test_set = IMDB(vocab_size, train = False, max_length = 400, path = data_path, labels_only = True, padding = False)
    dataloader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 0, collate_fn=pad_collate)
elif args.model_type == 'CNN':
    model_path = './Files/trained_bb_models/RNN_model_CNN.pt'
    test_set = IMDB(vocab_size, train = False, max_length = 400, path = data_path, labels_only = True, padding = True)
    dataloader = DataLoader(test_set, batch_size = 1, shuffle = False, num_workers = 0)


Explainer = IMDB_Explainer_Binary(model_path = model_path, m = m, baseline = baseline, model_type = args.model_type, m = m)


#######################
# Dataset Iterator

if args.verbose:
    batch_iterator = tqdm(enumerate(dataloader), total = max_index)
else:
    batch_iterator = enumerate(dataloader)


# initialize variables
x_list = []
label_list = []
unary_list = []
matrix_list = []
time_list = []

db_ind = {}

for idx, batch in batch_iterator:

    # advance batch iterator
    if idx < min_index:
        continue
    elif idx == max_index:
        break

    time_start = time.time()
    x = tensor2numpy(batch[0])
    y = tensor2numpy(batch[1])
    
    if x.shape[0] != 1: raise ValueError('batch size should be 1')
    shapley_values, shapley_matrix_pairwise = Explainer(x)
    
    # save individual shapley
    time_list.append(time.time() - time_start)
    x_list.append(x[0, :].tolist())
    label_list.append(y[0].item())
    unary_list.append(shapley_values)
    matrix_list.append(shapley_matrix_pairwise)


    if idx % 5 == 0:
        if not args.verbose:
            print('=====================')
            print('samples:' + str(idx+1))
            print('time per sample: ' + str(np.array(time_list).mean()))
        '''
        db_ind['x_list'] = x_list
        db_ind['label_list'] = label_list
        db_ind['unary_list'] = unary_list
        db_ind['matrix_list'] = matrix_list
        db_ind['time'] = time_list
        save_dict(db_ind, os.path.join(save_path, '%s-%s_checkpoint.pkl' % (str(min_index), str(max_index-1))))
        '''

db_ind['x_list'] = x_list
db_ind['label_list'] = label_list
db_ind['unary_list'] = unary_list
db_ind['matrix_list'] = matrix_list
db_ind['time_list'] = time_list
save_dict(db_ind, os.path.join(save_path, '%s-%s.pkl' % (str(min_index), str(max_index-1))))
#os.remove(os.path.join(save_path, '%s-%s_checkpoint.pkl' % (str(min_index), str(max_index-1))))
print('done!')


