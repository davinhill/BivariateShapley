
import os
import sys
from pathlib import Path

path = Path(os.path.abspath(__file__)).parents[2]
os.chdir(path)
sys.path.append('./BivariateShapley')

from utils_shapley import *
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch


def run_test(test, dataset, baseline, eval_test, matrix_input = False, eval_metric = 'PostHoc_accy', **kwargs):

    #################################
    #### Run Tests
    #################################
    '''
    args:
        test: test object
        dataset: string, dataset name
        baseline: string, baseline name
        eval_test: string, test name
        eval_metric: string, eval metric name for df_G test
    
    '''
    tmp = []
    df_tmp = []
    n_samples = test.n_samples
    baseline_accy = test.calc_baseline_accy()
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = 'cpu'

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if eval_test == 'ranking':
        abs_method = 'softplus'
        normalize = False
        personalize = kwargs.get('personalize',False)
        dmp = 0.85
        pct_mask = np.arange(0,1.04,0.04).tolist()
        accy, accy_PH = test.calc_ranking(pct_mask = pct_mask, normalize = normalize, abs_method = abs_method, metric = eval_metric, matrix_input = matrix_input, personalize = personalize, dmp = dmp) 

        if type(accy) != list: accy, accy_PH = [accy], [accy_PH] # if output is a scalar
        for i in range(len(accy)):
            tmp_ = [
            dataset,
            n_samples,
            np.array(test.time_list).mean(),
            baseline,
            normalize, # normalize
            abs_method,
            personalize,
            dmp,
            eval_metric,
            test.metrics['G_n_masked'][i] / test.metrics['G_n_feat'][i],
            baseline_accy,
            1.0,
            accy[i],
            accy_PH[i],
            ]
            tmp.append(tmp_)

        colnames = [
            'dataset',
            'n_samples',
            'time_per_sample',
            'baseline',
            'normalize',
            'abs_method',
            'personalize',
            'dmp',
            'eval_metric',
            'pct_masked',
            'baseline_accy',
            'baseline_accy_PH',
            'accy',
            'accy_PH',
        ]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif eval_test == 'MR':
        H = 'redundant'
        z = np.arange(0,10,0.2)
        z = 0.0001*(np.exp(z)-1)
        z = [1e-5, 1e-4, 1e-3, 1e-2]
        for gamma in z:
            pct_list = np.arange(0, 1.1, 0.1)
            for keep_perc in pct_list:

                MR, MR_PH = test.calc_MR(zero_threshold = gamma, H_matrix = H, keep = keep_perc)

                tmp_ = [
                    dataset,
                    n_samples,
                    baseline,
                    gamma,
                    H,
                    keep_perc,
                    baseline_accy,
                    1.0,
                    MR,
                    MR_PH,
                    sum(test.metrics['MR_n_masked']) / sum(test.metrics['MR_n_feat'])
                ]
                tmp.append(tmp_)

        colnames = [
            'dataset',
            'n_samples',
            'baseline',
            'gamma',
            'H_matrix',
            'keep_pct',
            'baseline_accy',
            'baseline_accy_PH',
            'MR',
            'MR_PH',
            'MR_pct_masked'
        ]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif eval_test == 'AR':
        H = 'redundant'
        z = np.arange(0,10,0.2)
        z = 0.0001*(np.exp(z)-1)
        z = z.tolist()
        z = z+[0.00001, 0.00002]
        for gamma in z:
            accy, accy_PH = test.calc_AR(zero_threshold = gamma, H_matrix = H, scale_MR = True)
            AR1, AR2, AR3, AR4, ARMR = accy
            AR1_PH, AR2_PH, AR3_PH, AR4_PH, ARMR_PH = accy_PH

            tmp_ = [
                dataset,
                n_samples,
                baseline,
                gamma,
                H,
                True, # scale_MR
                baseline_accy,
                1.0,
                AR1,
                AR2,
                AR3,
                AR4,
                ARMR,
                AR1_PH,
                AR2_PH,
                AR3_PH,
                AR4_PH,
                ARMR_PH,
                sum(test.metrics['AR1_n_masked']) / sum(test.metrics['AR1_n_feat']),
                sum(test.metrics['AR2_n_masked']) / sum(test.metrics['AR2_n_feat']),
                sum(test.metrics['AR3_n_masked']) / sum(test.metrics['AR3_n_feat']),
                sum(test.metrics['AR4_n_masked']) / sum(test.metrics['AR4_n_feat']),
                sum(test.metrics['ARMR_n_masked']) / sum(test.metrics['ARMR_n_feat'])
            ]
            tmp.append(tmp_)

        colnames = [
            'dataset',
            'n_samples',
            'baseline',
            'gamma',
            'H_matrix',
            'Scale_MR',
            'baseline_accy',
            'baseline_accy_PH',
            'AR1',
            'AR2',
            'AR3',
            'AR4',
            'ARMR',
            'AR1_PH',
            'AR2_PH',
            'AR3_PH',
            'AR4_PH',
            'ARMR_PH',
            'AR1_pct_masked',
            'AR2_pct_masked',
            'AR3_pct_masked',
            'AR4_pct_masked',
            'ARMR_pct_masked'
        ]


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif eval_test == 'centrality':
        for abs_method in ['relu', 'sigmoid', 'softplus', 'abs', 'none']:
            for degree in [1, 3, 5, 7, 9, 11]:
                #Note that Transpose should be True
                pct_mask = np.arange(0,1.0,0.04).tolist()
                accy_shapley, accy_shapley_PH = test.calc_Centralize(pct_mask = pct_mask, normalize = False, abs_method = abs_method, transpose = True, degree = degree) 

                for i, pct in enumerate(pct_mask):
                    tmp_ = [
                    dataset,
                    n_samples,
                    baseline,
                    False, # normalize
                    abs_method,
                    degree,
                    pct_mask[i],
                    baseline_accy,
                    1.0,
                    accy_shapley[i],
                    accy_shapley_PH[i],
                    ]
                    tmp.append(tmp_)

        colnames = [
            'dataset',
            'n_samples',
            'baseline',
            'normalize',
            'abs_method',
            'degree',
            'pct_masked',
            'baseline_accy',
            'baseline_accy_PH',
            'Cent_accy',
            'Cent_accy_PH',
        ]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    elif eval_test == 'density':
        for global_flag in [True, False]:
            z = np.arange(0,10,0.2)
            z = 0.0001*(np.exp(z)-1)
            gamma_list = z.tolist()
            density = test.calc_Density(gamma_list, global_flag = global_flag)
            for i, gamma in enumerate(density):
                tmp_ = [
                    dataset,
                    n_samples,
                    baseline,
                    global_flag,
                    gamma_list[i],
                    density[i]
                ]
                tmp.append(tmp_)

        colnames = [
            'dataset',
            'n_samples',
            'baseline',
            'global',
            'gamma',
            'density',
        ]

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df_tmp.extend(tmp)
    return pd.DataFrame(df_tmp, columns = colnames)



class Redundancy_Test():

    '''
    These functions take the calculated lists of univariate shapley and bivariate shapley attributions as input.
    calc_MR
    calc_AR
    calc_GRanking: converts attribution matrix into a ranking using PageRank. Then calculates accuracy after masking k values.
        This also calculates accuracy after masking topk univariate attribution values.
    
    
    '''
    def __init__(self, folder_path, value_function, verbose = True, rnn = False, mask_baseline = 0, **kwargs):
        
        self.value_function = value_function
        self.rnn = rnn
        self.metrics = {}
        self.verbose = verbose
        self.mask_baseline = mask_baseline
        self.AUC_baseline = kwargs.get('AUC_baseline')

        
        # load folder contents
        if verbose: print('Warning -- this script will load all pkl files in the folder:')
        fnames = os.listdir(folder_path)
        pkl_files = [file for file in fnames if file[-4:] == '.pkl']
        if verbose:
            print(pkl_files)
            print('loading...')

        
        # load files
        self.x_list = []
        self.label_list = []
        self.unary_list = []
        self.matrix_list = []
        self.time_list = []
        
        for file in pkl_files:
            db = load_dict(os.path.join(folder_path, file))
            self.x_list.extend(db['x_list'])
            self.label_list.extend(db['label_list'])
            self.unary_list.extend(db['unary_list'])
            self.matrix_list.extend(db['matrix_list'])
            self.time_list.extend(db['time_list'])
        
        '''
        ############### Image Problem Fix
        import pdb; pdb.set_trace()
        if not self.rnn and len(np.array(self.x_list[0]).shape)==3:
            for idx, image in enumerate(self.x_list):
                self.x_list[idx] = np.expand_dims(np.array(image, dtype = 'float32'), 0)
        #############################
        '''

        if verbose: print('done! ' + str(len(self.x_list)) + ' samples loaded.')
    
        # superpixels
        if self.rnn:
            self.num_superpixels = 0
        elif len(self.x_list[0].reshape(-1)) != len(self.unary_list[0].reshape(-1)):
            self.num_superpixels = len(self.unary_list[0].reshape(-1))
        else:
            self.num_superpixels = 0
            
        self.n_samples = len(self.x_list)

        # calculate x_list predictions
        if self.rnn:
            x_ = [np.array(tkn_list, dtype = 'int').reshape(1, -1) for tkn_list in self.x_list]
        else:
            x_ = np.concatenate(self.x_list, axis = 0)
        _, self.baseline_pred  = self.value_function.forward(x_)

    def calc_baseline_accy(self):
        '''
        calculate baseline accuracy
        '''
        if self.rnn:
            x_ = [np.array(tkn_list, dtype = 'int').reshape(1, -1) for tkn_list in self.x_list]
        else:
            x_ = np.concatenate(self.x_list, axis = 0)

        accy = self.value_function.eval_accy(x_, np.array(self.label_list))
        return accy
    
            
    def calc_MR(self, zero_threshold, H_matrix = 'redundant', keep = 0.1):
        '''
        H_matrix: 'redundant' or 'ratio'
        x_MR: Keep 1, Mask Everything Else
        
        '''
        if (keep >1) or (keep <0): raise ValueError('keep variable should be between 0 and 1')            

        x_MR = []
        self.metrics['MR_n_masked'] = []
        self.metrics['MR_n_feat'] = []
        self.metrics['MR_n_groups'] = []
        self.metrics['MR_masks'] = []
        self.metrics['MR_clusters'] = []
        
        for x, shapley_values, phi_plus in zip(self.x_list, self.unary_list, self.matrix_list):
            
            # x should be either (1 x d) or (1 x c x h x w) np array
            if self.rnn: x = np.array(x).reshape(1, -1)
            
            # calculate MR
            if H_matrix == 'redundant':
                phi_redundant = get_phi_redundant(phi_plus, zero_threshold)
            else:
                phi_redundant = get_phi_ratio(shapley_values, phi_plus, zero_threshold)

            MR = find_MR(phi_redundant)
            
            # initialize mask
            if self.num_superpixels > 0:
                mask = np.ones((1, phi_plus.shape[0]), dtype = 'double')
            else:
                mask = np.ones_like(x.reshape(1, -1), dtype = 'double')
                    
            self.metrics['MR_clusters'].append(MR)
            #~~~~~~~~~~~~~~~~~~~~~~        
            for node_group in MR:
                # node_group is a np array of feature indices
                
                # initialize mask for each node group
                mask_ = np.ones_like(mask)
                
                mask_[:, node_group] = 0

                keep_n = min(int(keep*len(node_group)), len(node_group))
                if keep_n >0:
                    keep_ind = np.random.choice(node_group, size = keep_n, replace = False)
                    mask_[:, keep_ind] = 1  # mask is 1 for features to keep, 0 for features to mask
    
                mask = np.multiply(mask, mask_) # accumulate masks
            #~~~~~~~~~~~~~~~~~~~~~~
                      
            # Reshape mask to fit data
            if self.num_superpixels > 0:
                mask, _ = superpixel_to_mask(numpy2cuda(mask), numpy2cuda(x))
                mask = tensor2numpy(mask)
            else:
                mask = mask.reshape(x.shape)

            # metrics
            self.metrics['MR_n_masked'].append((1-mask).reshape(-1).sum())
            self.metrics['MR_n_groups'].append(len(MR))
            self.metrics['MR_n_feat'].append(mask.size)  # percent of features masked
  

            # apply mask
            x_ = np.multiply(x, mask) + (1-mask) * self.mask_baseline
            self.metrics['MR_masks'].append(mask)
    
            x_MR.append(x_)
            
        # convert list to np array
        if self.rnn:
            x_MR_ = x_MR
        else:
            x_MR_ = np.concatenate(x_MR, axis = 0)
 
        accy = self.value_function.eval_accy(x_MR_, np.array(self.label_list))
        accy_PH = self.value_function.eval_accy(x_MR_, self.baseline_pred) # Post-hoc accuracy
        return accy, accy_PH
        

    def calc_AR(self, zero_threshold, H_matrix = 'redundant', keep = 0.0, calc_freq = False, scale_MR = False, transpose = False):
        '''
        
        H_matrix: 'redundant' or 'ratio'
        keep: % of features to keep for ARMR calculation

        x_AR1: Mask Source
        x_AR2: Mask Everything ex. Source
        
        x_AR3: Mask Sinks
        x_AR4: Mask Everything ex. Sinks
        '''
        
        if (keep >1) or (keep <0): raise ValueError('keep variable should be between 0 and 1')            

        x_AR1 = []
        x_AR2 = []
        x_AR3 = []
        x_AR4 = []
        x_ARMR = []
        
        self.metrics['AR1_n_masked'] = []
        self.metrics['AR2_n_masked'] = []
        self.metrics['AR3_n_masked'] = []
        self.metrics['AR4_n_masked'] = []
        self.metrics['ARMR_n_masked'] = []
        
        self.metrics['AR1_n_feat'] = []
        self.metrics['AR2_n_feat'] = []
        self.metrics['AR3_n_feat'] = []
        self.metrics['AR4_n_feat'] = []
        self.metrics['ARMR_n_feat'] = []
        
        self.metrics['AR1_masks'] = []
        self.metrics['AR2_masks'] = []
        self.metrics['AR3_masks'] = []
        self.metrics['AR4_masks'] = []
        self.metrics['ARMR_masks'] = []

        if calc_freq: self.metrics['AR_freq_matrix'] = []
        
        for x, shapley_values, phi_plus in zip(self.x_list, self.unary_list, self.matrix_list):
            
            # x should be either (1 x d) or (1 x c x h x w) np array
            if self.rnn: x = np.array(x).reshape(1, -1)
            
            if transpose:
                # Get H from G^T matrix
                if H_matrix == 'redundant':
                    phi_redundant = get_phi_redundant(phi_plus, zero_threshold)
                else:
                    phi_redundant = get_phi_ratio(shapley_values, phi_plus, zero_threshold)
            else:
                # Get H directly from G matrix
                if H_matrix == 'redundant':
                    phi_redundant = get_phi_redundant(phi_plus, zero_threshold, transpose = False)
                else:
                    phi_redundant = get_phi_ratio(shapley_values, phi_plus, zero_threshold, transpose=False)
                

            AR_source, AR_sink, freq_matrix = find_AR(phi_redundant, calc_freq)
            '''
            if phi_redundant.sum() == 0 or phi_redundant.sum() == phi_redundant.shape[0] **2 - phi_redundant.shape[0]:
                # if H has no connections (no redundancy) or fully connected (everything is redundant; there are no sources / sinks)
                #AR_source, AR_sink = [[], [], []], [[], [], []]
            else:
                try:
                    AR_source, AR_sink, freq_matrix = find_AR(phi_redundant, calc_freq)
                except ValueError:
                    import pdb; pdb.set_trace()
                    find_AR(phi_redundant, calc_freq)
            '''

            if calc_freq: self.metrics['AR_freq_matrix'].append(freq_matrix)

            # AR_source and AR_sink are lists
            # index 0: all nodes (list)
            # index 1: singular node sinks/sources (list)
            # index 2: groups of multiple node sinks/sources (list)
            
            # Sources ====================================
            
            # initialize mask
            if self.num_superpixels > 0:
                mask = np.ones((1, phi_plus.shape[0]), dtype = 'double')
            else:
                mask = np.ones_like(x.reshape(1, -1), dtype = 'double')           

            mask[:, AR_source[0]] = 0   
            mask = 1-mask       # mask is 1 if source, otherwise 0
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ARMR (mask Mutually Redundant Source Nodes at random (see MR calculation)) 

            pct_keep = (mask).reshape(-1).sum() / (mask.size)   # number of features masked

            mask_ARMR = np.ones_like(mask)
            if len(AR_source[2])>0:
                for node_group in AR_source[2]:
                    # node_group is a np array of feature indices
                    
                    # initialize mask for each node group
                    mask_ = np.ones_like(mask)
                    
                    mask_[:, node_group] = 0

                    if scale_MR:
                        keep_n = min(int(keep*len(node_group)), len(node_group))
                    else:
                        keep_n = min(int(pct_keep*len(node_group)), len(node_group))

                    keep_n = max(1, keep_n) # keep at least 1
                    if keep_n >0:
                        keep_ind = np.random.choice(node_group, size = keep_n, replace = False)
                        mask_[:, keep_ind] = 1  # mask is 1 for features to keep, 0 for features to mask
        
                    mask_ARMR = np.multiply(mask_ARMR, mask_) # accumulate masks
            
            mask_ARMR = np.multiply(mask, mask_ARMR)

            
            # Reshape mask to fit data
            if self.num_superpixels > 0:
                mask, _ = superpixel_to_mask(numpy2cuda(mask), numpy2cuda(x))
                mask = tensor2numpy(mask)

                mask_ARMR, _ = superpixel_to_mask(numpy2cuda(mask_ARMR), numpy2cuda(x))
                mask_ARMR = tensor2numpy(mask_ARMR)
            else:
                mask = mask.reshape(x.shape)
                mask_ARMR = mask_ARMR.reshape(x.shape)
            
            # metrics
            self.metrics['AR1_n_masked'].append((1-mask).reshape(-1).sum())  # number of features masked
            self.metrics['AR1_n_feat'].append(mask.size)  # percent of features masked
            
            # apply mask
            x_ = np.multiply(x, mask) + (1-mask) * self.mask_baseline
            self.metrics['AR1_masks'].append(mask)
            
            x_AR1.append(x_)
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # AR2 (Reverse of AR1)
            x_ = np.multiply(x, 1-mask) + (mask) * self.mask_baseline
            self.metrics['AR2_masks'].append(1-mask)
            x_AR2.append(x_)
        
            # metrics
            self.metrics['AR2_n_masked'].append((mask).reshape(-1).sum())  # number of features masked
            self.metrics['AR2_n_feat'].append(mask.size)  # percent of features masked

            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # ARMR 
            x_ = np.multiply(x, mask_ARMR)+ (1-mask_ARMR) * self.mask_baseline
            self.metrics['ARMR_masks'].append(mask_ARMR)
            x_ARMR.append(x_)
        
            # metrics
            self.metrics['ARMR_n_masked'].append((1-mask_ARMR).reshape(-1).sum())  # number of features masked
            self.metrics['ARMR_n_feat'].append(mask_ARMR.size)  # percent of features masked

            # Sinks ====================================
            
            # initialize mask
            if self.num_superpixels > 0:
                mask = np.ones((1, phi_plus.shape[0]), dtype = 'double')
            else:
                mask = np.ones_like(x.reshape(1, -1), dtype = 'double')           
             
            if AR_sink[0] is not None: mask[:, AR_sink[0]] = 0    # mask is 0 if sink, otherwise 1
            

        
            # Reshape mask to fit data
            if self.num_superpixels > 0:
                mask, _ = superpixel_to_mask(numpy2cuda(mask), numpy2cuda(x))
                mask = tensor2numpy(mask)
            else:
                mask = mask.reshape(x.shape)
                
            # metrics
            self.metrics['AR3_n_masked'].append((1-mask).reshape(-1).sum())
            self.metrics['AR3_n_feat'].append(mask.size)
            
            # apply mask
            x_ = np.multiply(x, mask) + (1-mask) * self.mask_baseline 
            self.metrics['AR3_masks'].append(mask)
            x_AR3.append(x_)
            
            
            #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # AR4 (Reverse of AR3)
            x_ = np.multiply(x, 1-mask)+ (mask) * self.mask_baseline
            self.metrics['AR4_masks'].append(1-mask)
            x_AR4.append(x_)
        
            # metrics
            self.metrics['AR4_n_masked'].append((mask).reshape(-1).sum())  # number of features masked
            self.metrics['AR4_n_feat'].append(mask.size)  # percent of features masked
           
            
        # Accy ====================================
        # convert list to np array
        if self.rnn:
            x_AR1_, x_AR2_, x_AR3_, x_AR4_, x_ARMR_ = x_AR1, x_AR2, x_AR3, x_AR4, x_ARMR
        else:
            x_AR1_ = np.concatenate(x_AR1, axis = 0)
            x_AR2_ = np.concatenate(x_AR2, axis = 0)
            x_AR3_ = np.concatenate(x_AR3, axis = 0)
            x_AR4_ = np.concatenate(x_AR4, axis = 0)
            x_ARMR_ = np.concatenate(x_ARMR, axis = 0)

        accy_AR1 = self.value_function.eval_accy(x_AR1_, np.array(self.label_list))
        accy_AR2 = self.value_function.eval_accy(x_AR2_, np.array(self.label_list))
        accy_AR3 = self.value_function.eval_accy(x_AR3_, np.array(self.label_list))
        accy_AR4 = self.value_function.eval_accy(x_AR4_, np.array(self.label_list))
        accy_ARMR = self.value_function.eval_accy(x_ARMR_, np.array(self.label_list))

        accy_AR1_PH = self.value_function.eval_accy(x_AR1_, self.baseline_pred)
        accy_AR2_PH = self.value_function.eval_accy(x_AR2_, self.baseline_pred)
        accy_AR3_PH = self.value_function.eval_accy(x_AR3_, self.baseline_pred)
        accy_AR4_PH = self.value_function.eval_accy(x_AR4_, self.baseline_pred)
        accy_ARMR_PH = self.value_function.eval_accy(x_ARMR_, self.baseline_pred)

        return [accy_AR1, accy_AR2, accy_AR3, accy_AR4, accy_ARMR], [accy_AR1_PH, accy_AR2_PH, accy_AR3_PH, accy_AR4_PH, accy_ARMR_PH]

    def calc_PR_scores(self, normalize = False, dmp = 0.85, abs_method = 'softplus', transpose = False, personalize = True):
        '''
        args:
            pct_mask: list of values between 0 and 1, representing the % of features to mask
            abs_method: activation function applied to G before PageRank. Can be none, sigmoid, softplus, relu
            transpose: Whether to transpose G matrix
            personalize: personalize pagerank with uni shapley values
        '''

        self.metrics['G_n_subgraphs'] = []

        if self.verbose:
            batch_iterator = tqdm(zip(self.x_list, self.unary_list, self.matrix_list), total = len(self.x_list))
        else:
            batch_iterator = zip(self.x_list, self.unary_list, self.matrix_list)

        PR_scores = []
        for x, shapley_values, phi_plus in batch_iterator:
            
            # x should be either (1 x d) or (1 x c x h x w) np array
            if self.rnn: x = np.array(x).reshape(1, -1)
            
            if normalize:
                phi_plus_ = normalize_phi(shapley_values, phi_plus)
            else:
                phi_plus_ = phi_plus

            if transpose:
                phi_plus_ = phi_plus_.transpose()

            phi_plus_+= 1e-70  # ensure every node is connected

            if abs_method == 'none':
                phi_plus_ = phi_plus_
            elif abs_method == 'sigmoid':
                phi_plus_ = sigmoid(phi_plus_)
            elif abs_method == 'softplus':
                phi_plus_ = softplus(phi_plus_)
            elif abs_method == 'relu':
                phi_plus_ = relu(phi_plus_)
            elif abs_method == 'abs':
                phi_plus_ = np.abs(phi_plus_)

            # sort pagerank / shapley values
            if personalize:

                # ensure at least one positive personalization weight
                if (shapley_values > 0).sum() == 0:
                    shapley_values_ = shapley_values - shapley_values.max() + 1e-70
                else:
                    shapley_values_ = shapley_values

                node_list, score_list = Phi_PageRank(phi_plus_, dmp = dmp, shapley_values =shapley_values_)
            else:
                node_list, score_list = Phi_PageRank(phi_plus_, dmp = dmp)


            self.metrics['G_n_subgraphs'].append(len(score_list))
            PR_scores.append(score_list[0])

        self.metrics['G_score_list'] = PR_scores
        if self.verbose: print('PageRank Done!')
        return PR_scores

    def _eval_PHaccy(self, score_list, pct_mask, **kwargs):

        if type(pct_mask) != list: raise ValueError('pct_mask must be a list of values between 0 and 1')

        x_shapley = []

        if self.verbose:
            batch_iterator = tqdm(zip(self.x_list, score_list), total = len(self.x_list))
        else:
            batch_iterator = zip(self.x_list, score_list)


        self.metrics['G_n_masked'] = np.zeros(len(pct_mask))
        self.metrics['G_n_feat'] = np.zeros(len(pct_mask))
        for x, shapley_values in batch_iterator:
            
            # x should be either (1 x d) or (1 x c x h x w) np array
            if self.rnn: x = np.array(x).reshape(1, -1)
            if type(x) == torch.Tensor: x = tensor2numpy(x)
            
            idx_shapley = np.argsort(shapley_values)

            x_shapley_ = [] # masking lowest k shapley values for comparison
            for z,k in enumerate(pct_mask):

                n_feat = len(shapley_values)
                n_mask = min(int(n_feat * k), n_feat)  # number of features to mask (convert from %)
                self.metrics['G_n_feat'][z] += n_feat
                self.metrics['G_n_masked'][z] += n_mask
                # initialize mask
                if self.num_superpixels > 0:
                    mask_topk = np.ones((1, n_feat), dtype = 'double')
                else:
                    mask_topk = np.ones_like(x.reshape(1, -1), dtype = 'double')           
                mask_shapley = mask_topk.copy()

                # update mask
                if n_mask>0:
                    mask_shapley[:, idx_shapley[:n_mask]] = 0

                # Reshape mask to fit data
                if self.num_superpixels > 0:
                    mask_shapley, _ = superpixel_to_mask(numpy2cuda(mask_shapley), numpy2cuda(x))
                    mask_shapley = tensor2numpy(mask_shapley)
                else:
                    mask_shapley = mask_shapley.reshape(x.shape)

                # apply mask
                x_ = np.multiply(x, mask_shapley) + (1-mask_shapley) * self.mask_baseline
                x_shapley_.append(x_)


            x_shapley.append(x_shapley_)

        # Accy ====================================
        # convert list to np array

        accy_shapley_list = []

        accy_shapley_list_PH = []

        for i, k in enumerate(pct_mask):
            x_shapley_ = [x[i] for x in x_shapley]

            if not self.rnn:
                x_shapley_ = np.concatenate(x_shapley_, axis = 0)

            accy_shapley = self.value_function.eval_accy(x_shapley_, np.array(self.label_list))

            accy_shapley_PH = self.value_function.eval_accy(x_shapley_, self.baseline_pred)

            accy_shapley_list.append(accy_shapley)
            accy_shapley_list_PH.append(accy_shapley_PH)

        return accy_shapley_list, accy_shapley_list_PH

    def calc_ranking(self, matrix_input = True, metric = 'PostHoc_Accuracy', pct_mask = np.arange(0,1,0.05).tolist(), **kwargs):
        if matrix_input:
            score_list = self.calc_PR_scores(**kwargs)
        else:
            score_list = self.unary_list

        if metric == 'PostHoc_accy':
            return self._eval_PHaccy(score_list, pct_mask)
        elif metric == 'AUC':
            return self._eval_AUC(score_list, **kwargs)
        else:
            raise ValueError('invalid metric specified')

    def _eval_AUC(self, score_list, **kwargs):
        '''
        Calculates a feature ranking using PageRank on the G Matrix. Similar to calc_GRanking, but the output is AUC (insertion and deletion) introduced in Petsiuk et al

        args:
            abs_method: activation function applied to G before PageRank. Can be none, sigmoid, softplus, relu
            transpose: Whether to transpose G matrix
            personalize: personalize pagerank with uni shapley values
        '''

        x_shapley_d = []
        x_shapley_i = []
        x_baseline = []

        if self.verbose:
            batch_iterator = tqdm(zip(self.x_list,score_list), total = len(self.x_list))
        else:
            batch_iterator = zip(self.x_list,score_list)


        self.metrics['G_n_masked'] = [1]
        self.metrics['G_n_feat'] = [1]
        for x, shapley_values in batch_iterator:
            

            # x should be either (1 x d) or (1 x c x h x w) np array
            if self.rnn: x = np.array(x).reshape(1, -1)
            if type(x) == torch.Tensor: x = tensor2numpy(x)
            
            # sort scores / values
            idx_shapley = np.argsort(shapley_values)

            x_shapley_i_ = [] # univariate shapley, insert
            x_shapley_d_ = [] # univariate shapley, delete

            n_feat = len(shapley_values)
            for n_mask in range(n_feat):

                # initialize mask
                if self.num_superpixels > 0:
                    mask_topk = np.ones((1, n_feat), dtype = 'double')
                else:
                    mask_topk = np.ones_like(x.reshape(1, -1), dtype = 'double')           
                mask_shapley = mask_topk.copy()

                ####################
                # update mask
                if n_mask>0:
                    mask_shapley[:, idx_shapley[-n_mask:]] = 0

                ###################
                # Reshape mask to fit data
                if self.num_superpixels > 0:
                    mask_shapley, _ = superpixel_to_mask(numpy2cuda(mask_shapley), numpy2cuda(x))
                    mask_shapley = tensor2numpy(mask_shapley)
                else:
                    mask_shapley = mask_shapley.reshape(x.shape)

                # apply mask
                x_ = np.multiply(x, mask_shapley) + (1-mask_shapley) * self.mask_baseline # mask out important features
                x_shapley_d_.append(x_)

                x_ = np.multiply(x, 1-mask_shapley) + (mask_shapley * self.mask_baseline) # insert important features
                x_shapley_i_.append(x_)

            x_shapley_d.append(x_shapley_d_)
            x_shapley_i.append(x_shapley_i_)
            x_baseline.append(np.ones_like(mask_shapley) * self.mask_baseline) # baseline sample for each x (not necessarily the same size for all samples in NLP datasets)

        #######################################################################################
        #######################################################################################

        # Accy ====================================
        # convert list to np array
        shapley_dAUC = 0.0
        shapley_iAUC = 0.0
        self.metrics['pred_d'] = []
        self.metrics['pred_i'] = []
        self.metrics['a'] = []
        for i,x in enumerate(self.x_list):
            x_shapley_d_ = x_shapley_d[i]
            x_shapley_i_ = x_shapley_i[i]

            if not self.rnn:
                x_shapley_d_ = np.concatenate(x_shapley_d_, axis = 0)
                x_shapley_i_ = np.concatenate(x_shapley_i_, axis = 0)
            else:
                x_shapley_d_ = np.concatenate(x_shapley_d_)
                x_shapley_i_ = np.concatenate(x_shapley_i_)

            if self.rnn: x = np.array(x).reshape(1, -1)
            if type(x) == torch.Tensor: x = tensor2numpy(x)

            self.value_function.init_baseline(x)
            pred_x_shapley_d = self.value_function(x_shapley_d_)
            pred_x_shapley_i = self.value_function(x_shapley_i_)

            if self.AUC_baseline is None:
                pred_baseline = self.value_function(x_baseline[i]) # prediction for baseline value (should be the lowest possible model prediction)
            else:
                pred_baseline = self.AUC_baseline # fix issue with imbalanced models

            shapley_dAUC += np.maximum(pred_x_shapley_d - pred_baseline, np.zeros_like(pred_x_shapley_d)).sum()/(len(x_shapley_d[i])) 
            shapley_iAUC += np.maximum(pred_x_shapley_i - pred_baseline, np.zeros_like(pred_x_shapley_i)).sum()/(len(x_shapley_i[i])) 
            '''
            shapley_dAUC += np.maximum(pred_x_shapley_d, np.zeros_like(pred_x_shapley_d)).sum()/(len(x_shapley_d[i])) 
            shapley_iAUC += np.maximum(pred_x_shapley_i, np.zeros_like(pred_x_shapley_i)).sum()/(len(x_shapley_i[i])) 
            '''

        # average over # samples
        shapley_dAUC /= len(self.x_list)
        shapley_iAUC /= len(self.x_list)

        return shapley_dAUC, shapley_iAUC

    def calc_Centralize(self, pct_mask, normalize = False, abs_method = 'none', transpose = True, degree = 1):
        '''
        args:
            pct_mask: list of values between 0 and 1, representing the % of features to mask
            abs_method: activation function applied to G before PageRank. Can be none, sigmoid, softplus, relu
            transpose: Whether to transpose G matrix
            personalize: personalize pagerank with uni shapley values
        '''
        if type(pct_mask) != list: raise ValueError('pct_mask must be a list of values between 0 and 1')

        x_shapley = []
        self.metrics['G_central_list'] = []

        if self.verbose:
            batch_iterator = tqdm(zip(self.x_list, self.unary_list, self.matrix_list), total = len(self.x_list))
        else:
            batch_iterator = zip(self.x_list, self.unary_list, self.matrix_list)

        for x, shapley_values, phi_plus in batch_iterator:
            
            # x should be either (1 x d) or (1 x c x h x w) np array
            if self.rnn: x = np.array(x).reshape(1, -1)
            
            if normalize:
                phi_plus_ = normalize_phi(shapley_values, phi_plus)
            else:
                phi_plus_ = phi_plus

            if transpose:
                phi_plus_ = phi_plus_.transpose()

            if abs_method == 'none':
                phi_plus_ = phi_plus_
                shapley_values_ = shapley_values 
            elif abs_method == 'sigmoid':
                phi_plus_ = sigmoid(phi_plus_)
                shapley_values_ = sigmoid(shapley_values)
            elif abs_method == 'softplus':
                phi_plus_ = softplus(phi_plus_)
                shapley_values_ = softplus(shapley_values)
            elif abs_method == 'relu':
                phi_plus_ = relu(phi_plus_)
                shapley_values_ = relu(shapley_values)
            elif abs_method == 'abs':
                phi_plus_ = np.abs(phi_plus_)
                shapley_values_ = abs(shapley_values)

            # centralized shapley
            for d in range(degree):
                shapley_values_ = np.matmul(phi_plus_, shapley_values_)


            # sort scores / values
            idx_shapley = np.argsort(shapley_values_)

            self.metrics['G_central_list'].append(shapley_values_)
            x_shapley_ = [] # masking lowest k shapley values for comparison
            for k in pct_mask:

                n_feat = len(shapley_values)

                n_mask = min(int(n_feat * k), n_feat)  # number of features to mask (convert from %)

                # initialize mask
                if self.num_superpixels > 0:
                    mask_topk = np.ones((1, phi_plus.shape[0]), dtype = 'double')
                else:
                    mask_topk = np.ones_like(x.reshape(1, -1), dtype = 'double')           
                mask_shapley = mask_topk.copy()

                # update mask
                if n_mask>0:
                    mask_shapley[:, idx_shapley[:n_mask]] = 0

                # Reshape mask to fit data
                if self.num_superpixels > 0:
                    mask_shapley, _ = superpixel_to_mask(numpy2cuda(mask_shapley), numpy2cuda(x))
                    mask_shapley = tensor2numpy(mask_shapley)
                else:
                    mask_shapley = mask_shapley.reshape(x.shape)

                # apply mask
                x_ = np.multiply(x, mask_shapley) + (1-mask_shapley) * self.mask_baseline
                x_shapley_.append(x_)


            x_shapley.append(x_shapley_)

        if self.verbose: print('PageRank Done!')
        # Accy ====================================
        # convert list to np array

        accy_shapley_list = []
        accy_shapley_list_PH = []

        for i, k in enumerate(pct_mask):
            x_shapley_ = [x[i] for x in x_shapley]

            if not self.rnn:
                x_shapley_ = np.concatenate(x_shapley_, axis = 0)

            accy_shapley = self.value_function.eval_accy(x_shapley_, np.array(self.label_list))

            accy_shapley_PH = self.value_function.eval_accy(x_shapley_, self.baseline_pred)

            accy_shapley_list.append(accy_shapley)
            accy_shapley_list_PH.append(accy_shapley_PH)

        return accy_shapley_list, accy_shapley_list_PH
        
    def calc_Density(self, zero_threshold_list, global_flag = True, transpose = False):
        '''

        '''

        density_list = []
        if global_flag:
            G_global = np.array(self.matrix_list).mean(axis = 0)
            for zero_threshold in zero_threshold_list:
                if transpose:
                    # Get H from G^T matrix
                    phi_redundant = get_phi_redundant(G_global, zero_threshold)
                else:
                    phi_redundant = get_phi_redundant(G_global, zero_threshold, transpose = False)
            
                G = nx.from_numpy_matrix(phi_redundant , create_using=nx.DiGraph)
                density_list.append(nx.density(G))
        else:
            for zero_threshold in zero_threshold_list:
                density = 0
                for phi_plus in self.matrix_list:
                    if transpose:
                        # Get H from G^T matrix
                        phi_redundant = get_phi_redundant(phi_plus, zero_threshold)
                    else:
                        phi_redundant = get_phi_redundant(phi_plus, zero_threshold, transpose = False)
                
                    G = nx.from_numpy_matrix(phi_redundant , create_using=nx.DiGraph)
                    density += nx.density(G)
                density_list.append(density / len(self.x_list))

        return density_list
            
