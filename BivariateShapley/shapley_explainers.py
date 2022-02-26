import sys
from shapley_sampling import Shapley_Sampling
from shapley_value_functions import *


class IMDB_Explainer_Binary(Shapley_Sampling):
    '''
    Wrapper for Shapley_Sampling class
    '''

    def __init__(self, model_path = './IMDB/RNN_model.pt', model_type = 'RNN', baseline = 'zero', exclude_list = [0, 2, 3], method = 'pairwise', **kwargs):
        '''
        args:
            model_path: path of pretrained model
            model_type: RNN or CNN (RNN requires length parameter)
            baseline: zero or one (zero represents 'p' padding token, one represents 'unknown' token)
            exclude list: list of feature values to exclude
                0 padding token
                2 start token
                3 end token
            
            see Shapley_Sampling for other optional parameters

        return: 
            unary shapley values, pairwise shapley matrix

        '''

        # load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        model = load_model(model_path)
        model.to(device)

        # initialize value function
        if model_type == 'RNN':
            self.value_function = eval_nlp_binary_rnn(model)
        elif model_type == 'CNN':
            self.value_function = eval_nlp_binary_cnn(model)
        else:
            raise ValueError('invalid model type') 

        # initialize shapley
        super().__init__(value_function = self.value_function, baseline = baseline, exclude_list = exclude_list, **kwargs)
        self.method = method


    def __call__(self, x):
        if self.method == 'pairwise':
            return self.pairwise_shapley_matrix(x, verbose = False)
        if self.method == 'interaction':
            return self.interaction_shapley(x, verbose = False)
        if self.method == 'shapleytaylor':
            return self.shapley_taylor(x, verbose = False)



class IMDB_Custom_Explainer(Shapley_Sampling):
    '''
    Wrapper for Shapley_Sampling class
    '''

    def __init__(self, exclude_list = [101, 102, 0], **kwargs):
        '''
        args:
            model_path: path of pretrained model
            model_type: RNN or CNN (RNN requires length parameter)
            baseline: zero or one (zero represents 'p' padding token, one represents 'unknown' token)
            exclude list: list of feature values to exclude
                0 padding token
                2 start token
                3 end token
            
            see Shapley_Sampling for other optional parameters

        return: 
            unary shapley values, pairwise shapley matrix

        '''
        self.value_function = eval_custom_transformer()

        # initialize shapley
        super().__init__(value_function = self.value_function, baseline = 'zero', exclude_list = exclude_list, **kwargs)

    def __call__(self, x):
        return self.pairwise_shapley_matrix(x)

class Image_Explainer(Shapley_Sampling):
    '''
    Wrapper for Shapley_Sampling class
    '''

    def __init__(self, binary_pred = True, model_path = '/MLP_baseline_binary_fullres.pt', baseline = 'zero', num_superpixels=0, sp_mapping = None, method = 'pairwise', **kwargs):
        '''
        args:
            binary_pred: boolean, whether the pytorch classifier is binary or multiclass
            model_path: path of pretrained model
            baseline: zero or mean 
            
            see Shapley_Sampling for other optional parameters

        return: 
            unary shapley values, pairwise shapley matrix

        '''

        # load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        model = load_model(model_path)
        model.to(device)

        # superpixel
        if num_superpixels == 0:
            sp_mapping = None
            self.value_function = eval_image(model, binary = binary_pred)
        elif num_superpixels < 0:
            raise ValueError('Number of Superpixels must be >= 0')
        else:
            if sp_mapping is None: raise ValueError('Missing SuperPixel mapping function')
            self.value_function = eval_image_superpixel(model, binary = binary_pred)

        # initialize shapley
        super().__init__(value_function = self.value_function, baseline = baseline, num_superpixels = num_superpixels, sp_mapping = sp_mapping, **kwargs)
        self.method = method


    def __call__(self, x):
        if self.method == 'pairwise':
            return self.pairwise_shapley_matrix(x, verbose = False)
        if self.method == 'interaction':
            return self.interaction_shapley(x, verbose = False)
        if self.method == 'shapleytaylor':
            return self.shapley_taylor(x, verbose = False)


class MLP_Explainer(Shapley_Sampling):
    '''
    Wrapper for Shapley_Sampling class
    '''

    def __init__(self, binary_pred = True, model_path = '/divorce/divorce_model.pt', baseline = 'mean', method = 'pairwise', **kwargs):
        '''
        args:
            binary_pred: boolean, whether the pytorch classifier is binary or multiclass
            model_path: path of pretrained model
            baseline: zero or mean 
            
            see Shapley_Sampling for other optional parameters

        return: 
            unary shapley values, pairwise shapley matrix

        '''

        # load model
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        model = load_model(model_path)
        model.to(device)

        self.value_function = eval_MLP(model, binary = binary_pred)
        self.method = method

        # initialize shapley
        super().__init__(value_function = self.value_function, baseline = baseline, **kwargs)


    def __call__(self, x):
        if self.method == 'pairwise':
            return self.pairwise_shapley_matrix(x, verbose = False)
        if self.method == 'interaction':
            return self.interaction_shapley(x, verbose = False)
        if self.method == 'shapleytaylor':
            return self.shapley_taylor(x, verbose = True)


class XGB_Explainer(Shapley_Sampling):
    '''
    Wrapper for Shapley_Sampling class
    '''

    def __init__(self, model_path = './census/model_census.json', baseline = 'zero',method = 'pairwise', **kwargs):
        '''
        args:
            binary_pred: boolean, whether the pytorch classifier is binary or multiclass
            model_path: path of pretrained model
            baseline: zero or mean 
            
            see Shapley_Sampling for other optional parameters

        return: 
            unary shapley values, pairwise shapley matrix

        '''

        # load model
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(model_path)
        self.value_function = eval_XGB(model)

        # initialize shapley
        super().__init__(value_function = self.value_function, baseline = baseline, **kwargs)
        self.method = method


    def __call__(self, x):
        if self.method == 'pairwise':
            return self.pairwise_shapley_matrix(x, verbose = False)
        if self.method == 'interaction':
            return self.interaction_shapley(x, verbose = False)
        if self.method == 'shapleytaylor':
            return self.shapley_taylor(x, verbose = False)


class RF_Explainer_binary(Shapley_Sampling):
    '''
    Wrapper for Shapley_Sampling class
    '''

    def __init__(self, model_path = '../drug/model_drug.pkl', baseline = 'zero', method = 'pairwise', **kwargs):
        '''
        args:
            binary_pred: boolean, whether the pytorch classifier is binary or multiclass
            model_path: path of pretrained model
            baseline: zero or mean 
            
            see Shapley_Sampling for other optional parameters

        return: 
            unary shapley values, pairwise shapley matrix

        '''

        # load model
        import pickle
        with open(model_path, 'rb') as fid:
            model = pickle.load(fid)
        self.value_function = eval_RF_binary(model)

        # initialize shapley
        super().__init__(value_function = self.value_function, baseline = baseline, **kwargs)

        self.method = method

    def __call__(self, x):
        if self.method == 'pairwise':
            return self.pairwise_shapley_matrix(x, verbose = False)
        if self.method == 'interaction':
            return self.interaction_shapley(x, verbose = False)
        if self.method == 'shapleytaylor':
            return self.shapley_taylor(x, verbose = False)

