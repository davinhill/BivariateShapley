import numpy as np
import os
import sys
os.chdir('../../../')
sys.path.append('./BivariateShapley')
from utils_shapley import *

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    

job_directory = './Tests/attribution_calculation/BivariateShapley/slurm_scripts'
script_path = './Tests/attribution_calculation/BivariateShapley/'
# Make top level directories
mkdir_p(job_directory)

m = {
    'drug': 500,
    'divorce': 500,
    'census': 500,
    'MNIST': 500,
    'CIFAR10': 500,
    'IMDB': 500,
    'COPD': 500,
}

samples_per_job = {
    'drug': 500,
    'divorce': 500,
    'census': 100,
    'MNIST': 50,
    'CIFAR10': 50,
    'IMDB': 50,
    'COPD': 5,
}

start_indices = {
    'drug': np.arange(0,500,samples_per_job['drug']).tolist(),
    'divorce': np.arange(0,500,samples_per_job['divorce']).tolist(),
    'census': np.arange(0,500,samples_per_job['census']).tolist(),
    'MNIST': np.arange(0,500,samples_per_job['MNIST']).tolist(),
    'CIFAR10': np.arange(0,500,samples_per_job['CIFAR10']).tolist(),
    'IMDB': np.arange(0,250,samples_per_job['IMDB']).tolist() + np.arange(12500,12750,samples_per_job['IMDB']).tolist(),
    'COPD': np.arange(0,500,samples_per_job['COPD']).tolist(),
}

gpu = {
    'drug': False,
    'divorce': False,
    'census': False,
    'MNIST': True,
    'CIFAR10': True,
    'IMDB': True,
    'COPD': False,
}

datasets = ['census', 'drug', 'divorce', 'IMDB', 'COPD', 'CIFAR10', 'MNIST']


for dataset in datasets:
    for start in start_indices[dataset]:
        job_file = os.path.join(job_directory,"%s_%s.job" %(dataset, start))
        python_script = 'iterate_%s.py' %dataset

        cmd = os.path.join(script_path, python_script)
        cmd = cmd + ' --dataset_min_index %s --dataset_samples %s --method_name %s --m %s' % (str(start), str(samples_per_job[dataset]), 'ShapleySampling', str(m[dataset]))
        submit_slurm(cmd, job_file, conda_env = 'shap', gpu = gpu[dataset], mem = 32)
