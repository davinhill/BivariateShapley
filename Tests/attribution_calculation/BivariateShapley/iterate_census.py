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

from argparse import ArgumentParser
from tqdm import tqdm
import time
import pandas as pd

############################################
# Define Test Parameters
############################################


parser = ArgumentParser(description='get phi plus matrices')

parser.add_argument('--m', type=int, default = 2000,
                    help='number of MC samples')

parser.add_argument('--verbose', action='store_true', default=False,
                    help='boolean, use tqdm')

parser.add_argument('--dataset_min_index', type = int,default=0,
                    help='iterate over dataset starting from min_index')

parser.add_argument('--dataset_samples', type = int,default=100,
                    help='number of samples, starting from min_index')

parser.add_argument('--baseline', type = str,default='fixed')
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



# parameters
method_name = args.method_name
model_path = './Files/trained_bb_models/model_census.json'
data_path = './Files/Data/census_x_test.pkl'
label_path ='./Files/Data/census_y_test.csv'
data_train_path = './Files/Data/census_x_train.pkl'

# Data Sample
dataset = pd.read_pickle(data_path)
dataset_train = pd.read_pickle(data_train_path)
labels = np.loadtxt(label_path)

if baseline == 'fixed':
    # fix baseline to be the average feature value in training set
    baseline = dataset_train.to_numpy().mean(axis = 0).reshape(1,-1)

from shapley_explainers import XGB_Explainer
# Initialize Explainer
Explainer = XGB_Explainer(model_path = model_path, baseline = baseline, dataset = dataset, m = m)
#######################
# Dataset Iterator


# initialize variables
x_list = []
label_list = []
unary_list = []
matrix_list = []
time_list = []

db_ind = {}
save_path = './Files/results_attribution/census_%s' % (method_name)
make_dir(save_path)


for idx in range(dataset.shape[0]):
 
    # advance batch iterator
    if idx < min_index:
        continue
    elif idx == max_index:
        break

    time_start = time.time()
    x = dataset.iloc[idx:(idx+1), :].to_numpy()
    label = labels[idx]
    
    if x.shape[0] != 1: raise ValueError('batch size should be 1')
    shapley_values, shapley_matrix_pairwise = Explainer(x)
    
    # save individual shapley
    time_list.append(time.time() - time_start)
    x_list.append(x)
    label_list.append(label)
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


