*Under Construction - Will be updated for ICML 2022*

# Bivariate Shapley

Implementation of the methods and experiments described in our spotlight [paper](https://openreview.net/forum?id=45Mr7LeKR9) at ICLR 2022:

Aria Masoomi, Davin Hill, Zhonghui Xu, Craig P. Hersh, Edwin K. Silverman, Peter J. Castaldi, Stratis Ioannidis, and Jennifer Dy. “Explanations of Black-Box Models Based on Directional Feature Interactions.”

<br />

![Image](https://github.com/davinhill/BivariateShapley/raw/main/Figures/fig1.jpg)


## Examples
We have two example implementations in the Examples folder. Both of these notebooks can be run in Google Colab.
* [Sentiment Analysis](https://github.com/davinhill/BivariateShapley/blob/main/Examples/example1_sentimentanalysis.ipynb)
* Census Dataset Classification *(under construction)*

## Bivariate Shapley Calculation
The Bivariate Shapley calculations are contained within the BivariateShapley folder.

**BivShap-S** (Shapley Sampling-based implementation): [shapley_sampling.py](https://github.com/davinhill/BivariateShapley/blob/main/BivariateShapley/shapley_sampling.py)

**BivShap-K** (KernelSHAP-based implementation): [shapley_kernel.py](https://github.com/davinhill/BivariateShapley/blob/main/BivariateShapley/shapley_kernel.py)




# Experiments

Below we detail the code used to evaluate Bivariate Shapley, as described in the Experiments section of the paper.


## Datasets and Black-Box Models
The black-box models evaluted in the experiments section are trained using the code in the BlackBox_Models folder. Datasets are not included in the repository due to file size, however most datasets are publically available (with the exception of COPDGene) with sources listed in the paper supplement. Please let us know if you have any issues with locating the datasets.

## Evaluation
Since calculating the G matrix can be time consuming for the number of samples we require, the tests are conducted in the following steps:
1. **Calculate G Matrices and/or univariate feature attribution values for all samples.** These values are calculated using the iterate_*.py scripts in the ./Tests/attribution_calculation folder, separated explanation method. Each iterate_*.py script iterates over all data samples specified and saves all explanations as lists in a dictionary, where each element of the list represents the explanation for a single sample. These iterate_*.py scripts were implemented in Slurm as separate jobs, in order to parallelize the calculations. The Slurm execution files are included for reference (execute_scripts.py), though these will likely need to be modified when implemented on a different system. Running the iterate_*.py script will save the associated dictionary in the ./Files/results_attribution folder.

2. **Evaluate accuracy metrics for the previously calculated feature attributions.** These scripts are located in the ./Tests/evaluation folder. These use the previously calculated feature attributions to generate the results listed in the experiments section of the paper. Execute the test_list.py file to calculate an evaluation metric for a particular method, specified using argparse. The related options are listed below. Running test_list.py will save a pandas dataframe in the ./Files/results_evaluation/ folder with the output of the experiment.


> **test_list.py options**
> 
> **--eval_test** (specifies which test to perform)
> * ranking: generates a ranking of features and evalutes the specified eval_metric
> * MR: finds and masks Mutually Redundant Features
> 
> **--eval_metric** (for the ranking test, specifies which metric to evalute)
> * PostHoc_accy: returns a list of posthoc accuracy scores for varying levels of masking
> * AUC: returns insertion and deletion AUC scores
> 
> **--dataset**
> * MNIST_196: MNIST with 196 superpixels
> * CIFAR10_255: CIFAR10 with 255 superpixels
> * IMDB
> * Census
> * Divorce
> * Drug
> 
> **--method** (the explanation method to evaluate)
> * BivariateShapley: Bivariate Shapley with Shapling Sampling implementation
> * BivariateShapley_kernel: Bivariate Shapley with kernelSHAP implementation
>
> Example: python test_list.py --eval_test ranking --eval_metric AUC --dataset Drug --method BivariateShapley_kernel


# Citation
```
@inproceedings{masoomi2022explanations,
  author    = {Aria Masoomi and
               Davin Hill and
               Zhonghui Xu and
               Craig P. Hersh and
               Edwin K. Silverman and
               Peter J. Castaldi and
               Stratis Ioannidis and
               Jennifer Dy},
  title     = {Explanations of Black-Box Models based on Directional Feature Interactions},
  booktitle = {10th International Conference on Learning Representations, {ICLR} 2022},
  publisher = {OpenReview.net},
  year      = {2022},
  url       = {https://openreview.net/forum?id=nIAxjsniDzg},
}
```
