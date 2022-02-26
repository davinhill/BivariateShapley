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

parser.add_argument('--num_superpixels', type=int, default = 255,
                    help='number of superpixels')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='boolean, use tqdm')

parser.add_argument('--dataset_min_index', type = int,default=0,
                    help='iterate over dataset starting from min_index')

parser.add_argument('--dataset_samples', type = int,default=100,
                    help='number of samples, starting from min_index')

parser.add_argument('--baseline', type = str,default='mean')
parser.add_argument('--method_name', type = str,default='BivariateShapley')
args = parser.parse_args()

############################################
# Define Tests
############################################

# Parameters
m = args.m
num_superpixels = args.num_superpixels
min_index = args.dataset_min_index
max_index = min_index + args.dataset_samples

baseline = args.baseline
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



# parameters
method_name = args.method_name
model_path = './Files/trained_bb_models/MLP_baseline_CIFAR10.pt'
data_path = './Files/Data/'
save_path = './Files/results_attribution/CIFAR10_%s_%s' % (str(args.num_superpixels), args.method_name)
sys.path.append('./BlackBox_Models/CIFAR10')
make_dir(save_path)
if num_superpixels >0:
    sp_mapping = superpixel_to_mask
else:
    sp_mapping = None


x_mean = torch.tensor([0.507, 0.487, 0.441])
x_std = torch.tensor([0.267, 0.256, 0.276])
UN = UnNormalize(x_mean, x_std)

data_transforms = {
    'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(x_mean, std=x_std)
    ])
}

# Data Sample
dataset = datasets.CIFAR10(
    root=data_path, train=False, download=True, transform=data_transforms['test'])
dataloader = DataLoader(dataset, batch_size = 1, shuffle = False, num_workers = 0)


from shapley_explainers import Image_Explainer

# Initialize Explainer
Explainer = Image_Explainer(binary_pred = False, model_path = model_path, baseline = baseline, num_superpixels= num_superpixels, sp_mapping = sp_mapping, dataset = dataset, m = m)
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

for idx, (x, label) in batch_iterator:

    # advance batch iterator
    if idx < min_index:
        continue
    elif idx == max_index:
        break

    time_start = time.time()
    if x.shape[0] != 1: raise ValueError('batch size should be 1')
    x = tensor2numpy(x)
    shapley_values, shapley_matrix_pairwise = Explainer(x)
    
    # save individual shapley
    time_list.append(time.time() - time_start)
    x_list.append(x)
    label_list.append(label[0].item())
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


