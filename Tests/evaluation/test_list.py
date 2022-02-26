import os
import sys
from pathlib import Path
path = Path(os.path.abspath(__file__)).parents[2]
os.chdir(path)
sys.path.append('./BivariateShapley')
sys.path.append('./Tests/evaluation')

import numpy as np
from datetime import datetime
import pandas as pd
from argparse import ArgumentParser

from utils_shapley import *
from test_functions import Redundancy_Test, run_test
from shapley_value_functions import *


#### Arguments
parser = ArgumentParser(description='specify test')
parser.add_argument('--eval_test', type=str, default = 'ranking', help='see test list file for options')
parser.add_argument('--eval_metric', type=str, default = 'PostHoc_accy', help='see test list file for options')
parser.add_argument('--dataset', type=str, default = 'MNIST_196', help='see test list file for options')
parser.add_argument('--method', type=str, default = 'BivariateShapley_kernel', help='see test list file for options')
args = parser.parse_args()


'''
options:

eval_test: specifies which test to perform
    ranking: generates a ranking of features and evalutes the specified eval_metric
    MR: finds and masks Mutually Redundant Features

eval_metric: for the ranking test, specifies which metric to evalute
    PostHoc_accy: returns a list of posthoc accuracy scores for varying levels of masking
    AUC: returns insertion and deletion AUC scores

dataset:
    MNIST_196: MNIST with 196 superpixels
    CIFAR10_255: CIFAR10 with 255 superpixels
    IMDB
    COPD
    Census
    Divorce
    Drug

method: the explanation method to evaluate
    BivariateShapley: Bivariate Shapley with Shapling Sampling implementation
    BivariateShapley_kernel: Bivariate Shapley with kernelSHAP implementation

'''



baseline = args.method

#### Initialize Variables
x_mean = torch.tensor([0.507, 0.487, 0.441])
x_std = torch.tensor([0.267, 0.256, 0.276])
unnormalize = UnNormalize(x_mean, x_std)
df_tmp = []
dataset = args.dataset
path = ('./Files/results_attribution/%s_%s' % (dataset, baseline))
save_name = args.method # string to differentiate save file
if args.eval_test != 'ranking': eval_metric = ''


#### Matrix Input
if args.method in ['BivariateShapley', 'BivariateShapley_kernel', 'intKS', 'excess', 'shapleytaylor']:
    matrix_input = True
else:
    matrix_input = False

#### Matrix Input
if args.eval_test == 'MR':
    eval_metric = ''
else:
    eval_metric = args.eval_metric

#### Bivariate-specific parameters
if args.method in ['BivariateShapley', 'BivariateShapley_kernel']:
    personalize = True
else:
    personalize = False


#### Datasets
if dataset == 'MNIST_196':
    sys.path.append('./BlackBox_Models/MNIST')
    value_function = eval_image(load_model('./Files/trained_bb_models/MLP_baseline.pt'), binary = False)
    test = Redundancy_Test(path, value_function, verbose = False)

elif dataset == 'CIFAR10_255':
    sys.path.append('./BlackBox_Models/CIFAR10')
    value_function = eval_image(load_model('./Files/trained_bb_models/MLP_baseline_CIFAR10.pt'), binary = False)
    test = Redundancy_Test(path, value_function, verbose = False)

elif dataset == 'divorce':
    sys.path.append('./BlackBox_Models/divorce')
    value_function = eval_MLP(load_model('./Files/trained_bb_models/divorce_model.pt'), binary = True)
    test = Redundancy_Test(path, value_function, mask_baseline= 3, verbose = False)

elif dataset == 'IMDB':
    sys.path.append('./BlackBox_Models/IMDB')
    value_function = eval_nlp_binary_rnn(load_model('./Files/trained_bb_models/RNN_model.pt'))
    test = Redundancy_Test(path, value_function, rnn=True, verbose = False)

elif dataset == 'COPD':
    sys.path.append('./BlackBox_Models/COPD')
    value_function = eval_MLP(load_model('./Files/trained_bb_models/COPD_model.pt'), binary = True)
    test = Redundancy_Test(path, value_function, verbose = False)

elif dataset == 'drug':
    import pickle
    with open('./Files/trained_bb_models/model_drug.pkl', 'rb') as fid:
        model = pickle.load(fid)
    value_function = eval_RF_binary(model)
    test = Redundancy_Test(path, value_function, verbose = False)

elif dataset == 'census':
    import xgboost as xgb
    model = xgb.Booster()
    model.load_model('./Files/trained_bb_models/model_census.json')
    # set baseline value to be the expectation of the marginal dist'n of each feature
    dataset_train = pd.read_pickle('./Files/Data/census_x_train.pkl')
    baseline_value = dataset_train.to_numpy().mean(axis = 0).reshape(1,-1)
    value_function = eval_XGB(model)
    test = Redundancy_Test(path, value_function, verbose = False, mask_baseline = baseline_value)



start = datetime.now()
print('===================')
print(dataset)
print(baseline)

##########
df_tmp = run_test(test, dataset, baseline, eval_test = args.eval_test, eval_metric = eval_metric, personalize = personalize, matrix_input = matrix_input)
print(str(datetime.now() - start))
df_tmp.to_pickle('./Files/results_evaluation/%s_%s_%s_%s.pkl' % (args.eval_test, dataset, save_name, eval_metric))

