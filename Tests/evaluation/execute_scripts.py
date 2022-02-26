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


  
datasets = ['census', 'drug', 'divorce', 'IMDB', 'COPD', 'CIFAR10_255', 'MNIST_196']
methods = ['BivariateShapley', 'BivariateShapley_kernel']
eval_tests = ['ranking', 'MR']
eval_metrics = ['PostHoc_accy', 'AUC']

for dataset in datasets:
    for method in methods:
        for eval_test in eval_tests:
            if eval_test == 'ranking':
                for eval_metric in eval_metrics:
                    job_file = os.path.join(job_directory,"%s_%s_%s_%s.job" %(dataset, method, eval_test, eval_metric))
                    python_script = 'test_list.py'
                    cmd = os.path.join(script_path, python_script)
                    cmd = cmd + ' --eval_test %s --eval_metric %s --dataset %s --method %s' % (eval_test, eval_metric, dataset, method)
                    submit_slurm(cmd, job_file, conda_env = 'shap', gpu = True, mem = 32, job_name = python_script[-11:-3])
            else:
                job_file = os.path.join(job_directory,"%s_%s_%s.job" %(dataset, method, eval_test))
                python_script = 'test_list.py'
                cmd = os.path.join(script_path, python_script)
                cmd = cmd + ' --eval_test %s --dataset %s --method %s' % (eval_test, dataset, method)
                submit_slurm(cmd, job_file, conda_env = 'shap', gpu = False, mem = 32, job_name = python_script[-11:-3])
            
