import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils_shapley import tensor2numpy, numpy2cuda, tensor2cuda, save_dict, find_redundancy
from tqdm import tqdm
import pandas as pd
import torch
import time


class Shapley_Sampling():

    def __init__(self, value_function, dataset=None, m = 1000, baseline = 'zero', batch_size = 128, exclude_list = [], num_superpixels = 0, sp_mapping = None):
        '''
        args:
            dataset: torchvision dataset object (if using baseline == 'mean', otherwise optional)
            value_function: function to be evaluated
            m: number of sampling iterations
            baseline: either 'mean' to sample w from dataset, or 'zero' to set w to zero.
            batch_size: number of permuted samples to pass through the value_function for one forward pass
            exclude_list: list of feature values to exclude when calculating shapley
            sp_mapping: superpixel decoder function 
        '''

        self.dataset = dataset
        if type(self.dataset) == pd.DataFrame:
            self.pandas_dataset = True
        else:
            self.pandas_dataset = False

        self.value_function = value_function
        self.m = m
        self.baseline = baseline
        self.batch_size = batch_size
        self.exclude_list = exclude_list
        self.sp_mapping = sp_mapping
        self.num_superpixels = num_superpixels
    
    def _pairwise_shapley(self, x, feat_i = 0, feat_fixed = None, feat_fixed_present = True):
        '''
        calculates pairwise shapley values for given feature i

        args:
            x: data sample to explain (numpy 1d vector)
            feat_i: index of feature to calculate shapley value for (int)

        return:
            2 values:
                Unary shapley values
                Pairwise shapley values for feat_i w.r.t. all other features (numpy 1d vector)
        '''

        # intialize input
        x = tensor2numpy(x)
        n_features = x.shape[1]
        phi_i = np.zeros((1, n_features)) # pairwise shapley vector

        # intialize dataloader
        if self.baseline == 'mean' and self.pandas_dataset == False:
            if self.dataset is None: raise ValueError('Missing a dataset to calculate Mean Baseline')
            self.dataloader = DataLoader(self.dataset, batch_size = x.shape[0], shuffle = True, num_workers = 0)
            self.dataiterator = iter(self.dataloader)

        # each sample in the batch is a different subset of the original x
        n_batches = int(np.ceil(self.m / self.batch_size))
        for i in range(n_batches):

            # initialize variables
            O_n = np.zeros((self.batch_size, n_features), dtype = 'int')
            x_ = np.zeros((self.batch_size, n_features))
            b = np.zeros((self.batch_size, n_features), dtype = 'int')
            O_inv_cons = []
            loc_i_cons = []

            # batch different subsets of x
            for j in range(self.batch_size):

                # Create feature permutation
                O_n[j, :] = np.random.permutation(n_features)  

                # Invert Permutation Mapping
                O_inv = []
                for k in range(n_features):
                    O_inv.append(np.where(O_n[j, :] == k)[0][0])
                O_inv_cons.append(O_inv)

                loc_i = np.where(O_n[j, :] == feat_i)[0][0] # permuted location for feature i
                loc_i_cons.append(loc_i)
                x_[j, :] = x[:,O_n[j, :]] # rearrange input x to match permutation

                # create binary mask of selected features
                b[j, :] = np.concatenate((np.ones_like(x)[:,0:loc_i],np.zeros_like(x)[:,loc_i:]), axis = 1)[:,O_inv] # binary mask of selected subset, 1 indicates presence
            
           ###################################
            # Sample from Baseline
            ###################################
            w = self._sample_baseline(np.repeat(x, self.batch_size, axis = 0))

            # initialize variables
            w_ = np.zeros((self.batch_size, n_features))
            b_1 = np.zeros((self.batch_size, n_features))
            b_2 = np.zeros((self.batch_size, n_features))

            # for each subset/permutation of x, rearrange baseline to match
            for j in range(self.batch_size):
                O_inv = O_inv_cons[j]
                loc_i = loc_i_cons[j]
                w_[j, :] = w[j,O_n[j, :]] # rearrange data sample w to match permutation
                
                b_1[j, :] = np.concatenate((x_[j,0:loc_i+1],w_[j,loc_i+1:])) # v(Si)
                b_2[j, :] = np.concatenate((x_[j,0:loc_i],w_[j,loc_i:]))  # v(S)

                # revert permutation
                b_1[j, :], b_2[j, :] = b_1[j,O_inv], b_2[j,O_inv]


                # if calculating interaction shapley with fixed features
                if feat_fixed is not None:
                    # if feature j is fixed to present, set it to original value. Else, set it to the baseline 
                    if feat_fixed_present:
                        b_1[j, feat_fixed], b_2[j, feat_fixed] = x[j, feat_fixed], x[j, feat_fixed]
                    else:
                        b_1[j, feat_fixed], b_2[j, feat_fixed] = w[j, feat_fixed], w[j, feat_fixed]

            ###################################
            # Calculate contribution
            ###################################
            contribution_i = self.value_function(b_1, w=w) - self.value_function(b_2, w=w)

            # sum contribution w.r.t. j
            b[:, feat_i] = 1 # fill in feature i in mask to calculate unary shapley
            contribution_i = np.repeat(np.expand_dims(contribution_i, axis = 1), n_features, axis =1) # expand contribution i to match dims
            phi_i += (np.multiply(b, contribution_i)).sum(axis = 0)

        phi_i = phi_i/(n_batches * self.batch_size) # pairwise shapley values
        shapley = phi_i[:, feat_i].copy() # shapley value for feature i
        phi_i[:, feat_i] = 0 # zero out diagonal values
        return shapley, phi_i


    def _sample_baseline(self, x):
        '''
        sample baseline given x conditioned on binary mask

        args:
            x: full data sample  n x d

        return:
            baseline sample w
        '''

        if type(self.baseline) == str:
            if self.baseline == 'mean':
                if self.pandas_dataset:
                    # pandas dataset
                    w = self.dataset.sample(n=x.shape[0]).to_numpy().reshape(x.shape[0], -1)
                else:
                    # pytorch dataloader
                    w = []
                    for i in range(x.shape[0]):
                        try:
                            tmp, _ = next(self.dataiterator)
                        except StopIteration:
                            dataiterator = iter(self.dataloader)
                            tmp, _ = next(dataiterator)

                        w.append(torch.flatten(tmp, 1))

                    w = torch.cat(w, dim = 0)
                    w = w[:x.shape[0], ...] # ensure number of baseline samples = number of x samples
                    w = tensor2numpy(w) # convert w to numpy array

            elif self.baseline == 'zero':
                # set w to zero
                w = np.zeros_like(x)

            elif self.baseline == 'null':
                # set w to be -1 (for use in binary samples like Glove)
                w = np.zeros_like(x)
                w = w - 1

            elif self.baseline == 'one':
                # set w to be 1 (for use in imdb, where 1 = unknown)
                w = np.ones_like(x)

            elif self.baseline == 'three':
                # set w to be 3 (for use in divorce, where 3 = average)
                w = np.ones_like(x)
                w = w + 2
        
        elif type(self.baseline) == np.ndarray:
            # w is a 1xd numpy array
            w = np.repeat(self.baseline, x.shape[0], axis = 0)

        return w



    def pairwise_shapley_matrix(self, x, verbose = True):
        '''
        iterates the pairwise_shapley function for all features in observation x

        args:
            x: data sample to explain (numpy 1d vector)

        return:
            numpy matrix of all pairwise shapley values (rows represent features i, columns represent fixed features j.)
            diagonal values are non-pairwise shapley values.
        '''

        # initialize evaluation function
        self.value_function.init_baseline(x = x, num_superpixels = self.num_superpixels, sp_mapping = self.sp_mapping)

        # if using superpixels
        if self.num_superpixels > 0:
            x = np.ones((1, self.num_superpixels))
        else:
            x = x.reshape(1, -1)  # input to shapley function must be 1 x d

        # initialize variables
        n_features = x.reshape(x.shape[0], -1).shape[1]
        shapley_matrix = np.zeros((n_features, n_features))
        shapley_values = np.zeros((n_features))

        # verbosity
        if verbose:
            batch_iterator = tqdm(range(n_features))
        else:
            batch_iterator = range(n_features)

        # iterate over all features
        if len(self.exclude_list) == 0:
            for i in batch_iterator:
                shapley_values[i], shapley_matrix[i, :] = self._pairwise_shapley(x, feat_i = i)

        # if there are features to be excluded
        else:
            for i in batch_iterator:
                if x[0, i] in self.exclude_list:  # if there are special tokens to exclude for shapley calculation
                    shapley_values[i], shapley_matrix[i,:] = 0, np.zeros((n_features))
                else:
                    shapley_values[i], shapley_matrix[i, :] = self._pairwise_shapley(x, feat_i = i)

        return shapley_values, shapley_matrix.transpose()
    
