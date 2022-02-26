import torch
import torch.nn.functional as F
from utils_shapley import *




class eval_Syn0():
    def __init__(self,c=5, **kwargs):
        self.c = c
        self.j = None
        self.i = None

    def init_baseline(self,x=np.ones((1,3)), c = 5,j = None, i = None, fixed_present = True, baseline_value = 0, **kwargs):
        self.x_baseline = x
        self.j = j
        self.i = i
        self.fixed_present = fixed_present
        self.baseline_value = baseline_value # if baseline is not zero
        
    def __call__(self, x, **kwargs):

        # Shapley Excess--------------------------------------
        # feature i and j are assumed to be in the same coalition, therefore j is present if i is present
        if self.i is not None:
            j_indicator = (self.x_baseline[:, self.i] == x[:,self.i]).reshape(-1,1)*1 # 1 if j should be present, 0 if j should be absent

            j_present = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(-1, 1)
            j_absent = (np.zeros((x.shape[0], self.x_baseline.shape[1])) + self.baseline_value)[:,self.j].reshape(-1,1)

            j_vector = j_indicator * j_present + (1-j_indicator) *  j_absent
            x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------


        # Interaction Shapley---------------------------------
        if (self.j is not None) and (self.i is None):                               #
            if self.fixed_present:                           #
                j_vector = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(x.shape[0], -1)
                x = np_insert(x, j_vector, index = self.j)   #
            else:                                            #
                j_vector = (np.zeros((x.shape[0], self.x_baseline.shape[1])) + self.baseline_value)[:,self.j].reshape(-1,1)
                x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------
        #return x.sum(axis = 1) + self.c * x.prod(axis = 1)
        #return 0.5*(x[:,0] + x[:,1])*(1-x[:,2]) + x[:,2]

        #return (((x == 1).max(axis = 1)*1 + (x == 0).max(axis = 1)*1 == 2)*1).reshape(-1)
        return 5*x[:,0] + x[:,1] + x[:,0] * x[:,1] - 8*x[:,3] - 2*x[:,3] * x[:,3]



class eval_glove():
    def __init__(self, **kwargs):
        pass

    def init_baseline(self, **kwargs):
        pass
        
    def __call__(self, x, **kwargs):
        n_0 = len(np.where(x == 0)[1]) # number of Left Gloves
        n_1 = len(np.where(x == 1)[1]) # number of Right Gloves

        if n_0 > 0 and n_1 > 0:
            return 1
        else:
            return 0

class eval_MLP():
    def __init__(self, model, binary = True, reshape = False):
        self.model = model
        self.model.eval()

        self.baseline = None
        self.binary = binary
        self.j = None
        self.i = None
        self.baseline_value = 0
        self.reshape = reshape



    def init_baseline(self, x, j = None, i = None, fixed_present = True, baseline_value = 0, **kwargs):
        '''
        set baseline prediction for original non-perturbed x value
        args:
            x: single sample. numpy array. 1 x d
        '''
        _, self.d = x.shape

        if self.binary:
            self.baseline = torch.sigmoid(self.model(numpy2cuda(x)))
        else:
            self.baseline = self.model(numpy2cuda(x)).argmax(dim = 1)
        
        self.x_baseline = x
        self.j = j
        self.i = i
        self.fixed_present = fixed_present
        self.baseline_value = baseline_value # if baseline is not zero

    def forward(self, x):
        '''
        forward pass of model, returns predictions.

        args:
            data: list of np arrays (note, this is different __call__)
        return:
            P(y|x) and predictions
        '''

        x = numpy2cuda(x)
        x = x.type(dtype = torch.float32)

        with torch.no_grad():
            output = self.model(x)

            if self.binary:
                pred = output >= 0.0
            else:
                pred = output.max(1, keepdim=True)[1]  # Calculate Predictions
        
        return tensor2numpy(output), tensor2numpy(pred)

    def eval_accy(self, x, label):
        '''
        given samples and labels, calculate accuracy

        args:
            x: np matrix
            label: np array of labels
        '''

        _, pred = self.forward(x)
        pred = numpy2cuda(pred)
        label = numpy2cuda(label)

        if self.binary:
            truth = label >= 0.5
            accy = pred.eq(truth).sum().item()
        else:
            accy = pred.eq(label.view_as(pred)).sum().item()

        return accy/len(label)

    def __call__(self, x, **kwargs):
        '''
        Note: The input to Shapley Function will be flattened. Therefore, it may be necessary to reshape x prior to a forward pass.
        '''
        if self.baseline is None: raise Exception('Need to first initialize baseline in evaluation function!')

        # Shapley Excess--------------------------------------
        # feature i and j are assumed to be in the same coalition, therefore j is present if i is present
        if self.i is not None:
            j_indicator = (self.x_baseline[:, self.i] == x[:,self.i]).reshape(-1,1)*1 # 1 if j should be present, 0 if j should be absent

            j_present = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(-1, 1)
            j_absent = (np.zeros((x.shape[0], self.x_baseline.shape[1])) + self.baseline_value)[:,self.j].reshape(-1,1)

            j_vector = j_indicator * j_present + (1-j_indicator) *  j_absent
            x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------

        # Interaction Shapley---------------------------------
        if (self.j is not None) and (self.i is None):                               #
            if self.fixed_present:                           #
                j_vector = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(x.shape[0], -1)
                x = np_insert(x, j_vector, index = self.j)   #
            else:                                            #
                j_vector = (np.zeros((x.shape[0], self.x_baseline.shape[1])) + self.baseline_value)[:,self.j].reshape(-1,1)
                x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------

        with torch.no_grad():

            x = numpy2cuda(x).type(dtype=torch.float32)
            pred = self.model(x)
            if self.binary:
                pred = torch.sigmoid(pred)

                if self.reshape:
                    output = tensor2numpy(pred).reshape(-1, 1)
                    output = np.concatenate((np.ones_like(output) - output, output), axis = 1)
                    return output      
                    
                if self.baseline < 0.5: pred = 1-pred
            else:
                pred = torch.exp(-F.cross_entropy(pred, self.baseline.expand(pred.shape[0]), reduction = 'none'))
        return tensor2numpy(pred)

class eval_nlp_binary_rnn():
    '''
    note: this is for a rnn that requires length parameter
    '''
    def __init__(self, model, reshape = False, **kwargs):
        self.model = model
        self.model.eval()

        self.baseline = None
        self.j = None
        self.i = None
        self.reshape = reshape

    def init_baseline(self, x, j = None, i=None, fixed_present = True, baseline_value = 0, **kwargs):
        '''
        set baseline prediction for original non-perturbed x value
        '''

        self.x_baseline = x
        self.j = j
        self.i = i
        self.fixed_present = fixed_present

        self.x_lens = torch.tensor(x.shape[1], dtype=torch.int64).reshape(1)
        self.baseline = torch.sigmoid(self.model(numpy2cuda(x), self.x_lens))
        self.baseline_value = baseline_value # if baseline is not zero

    def forward(self, data, data_lens = None):
        '''
        forward pass of model, returns predictions.

        args:
            data: list of np arrays (note, this is different __call__)
        return:
            P(y|x) and predictions
        '''
        if type(data) == list:
            x = []
            x_len = []
            for tkn_list in data:
                if len(tkn_list[0, :]) < 400:
                    tmp = np.concatenate((tkn_list[0, :], np.zeros(400 - len(tkn_list[0, :]), dtype = 'int')), axis = 0)
                    tmp_len = len(tkn_list[0, :])
                else:
                    tmp = tkn_list[0,:400]
                    tmp_len = 400
                x.append(tmp)
                x_len.append(tmp_len)
            x = np.array(x, dtype = 'int')
            x = numpy2cuda(x)
            x_len = list2cuda(x_len)
        elif type(data) == np.ndarray:
            x = numpy2cuda(data.astype('intc'))
            if data_lens is None:
                x_len = tensor2cuda(torch.zeros(x.shape[0], dtype = torch.int32)+400)
            else:
                x_len = numpy2cuda(data_lens)
        elif type(data) == torch.Tensor:
            x = tensor2cuda(data)
            if data_lens is None:
                x_len = tensor2cuda(torch.zeros(x.shape[0], dtype = torch.int32)+400)
            else:
                x_len = tensor2cuda(data_lens)



        with torch.no_grad():
            output = self.model(x, x_len)
            pred = output >= 0.0
        return tensor2numpy(torch.sigmoid(output)), tensor2numpy(pred)


    def eval_accy(self, data, label, data_lens = None):
        '''
        given samples and labels, calculate accuracy

        args:
            data: list of np arrays (note, this is different __call__)
            label: np array of labels
        '''

        _, pred = self.forward(data, data_lens)
        pred = numpy2cuda(pred)
        label = numpy2cuda(label)

        truth = label >= 0.5
        accy = pred.eq(truth).sum().item()
        return accy / len(label)


    def __call__(self, x, **kwargs):
        if self.baseline is None: raise Exception('Need to first initialize baseline in evaluation function!')

        with torch.no_grad():

            # Shapley Excess--------------------------------------
            # feature i and j are assumed to be in the same coalition, therefore j is present if i is present
            if self.i is not None:
                j_indicator = (self.x_baseline[:, self.i] == x[:,self.i]).reshape(-1,1)*1 # 1 if j should be present, 0 if j should be absent

                j_present = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(-1, 1)
                j_absent = (np.zeros((x.shape[0], self.x_baseline.shape[1])) + self.baseline_value)[:,self.j].reshape(-1,1)

                j_vector = j_indicator * j_present + (1-j_indicator) *  j_absent
                x = np_insert(x, j_vector, index = self.j)   #
            #-----------------------------------------------------

            # Interaction Shapley---------------------------------
            if (self.j is not None) and (self.i is None):                               #
                if self.fixed_present:                           #
                    j_vector = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(x.shape[0], -1)
                    x = np_insert(x, j_vector, index = self.j)   #
                else:                                            #
                    j_vector = np.zeros((x.shape[0], 1))         #
                    x = np_insert(x, j_vector, index = self.j)   #
            #-----------------------------------------------------

            x = numpy2cuda(x.astype('int'))
            #X_lens = torch.tensor(X.shape[1], dtype=torch.int64).reshape(1)
            pred = self.model(x, self.x_lens.expand(x.shape[0]))
            pred = torch.sigmoid(pred)

            if self.reshape:
                output = tensor2numpy(pred).reshape(-1, 1)
                output = np.concatenate((np.ones_like(output) - output, output), axis = 1)
                return output                  

            if self.baseline < 0.5: pred = 1-pred
        return pred.cpu().detach().numpy()


class eval_nlp_binary_cnn():
    ##### NEEDS TO BE UPDATED ####
    def __init__(self, model, max_length = 400, **kwargs):
        self.model = model
        self.model.eval()
        self.max_length = max_length
        self.baseline = None
        self.j = None
        self.i = None

    def init_baseline(self, x, j = None, fixed_present = True, baseline_value = 0, **kwargs):
        '''
        set baseline prediction for original non-perturbed x value
        '''
        self.x_baseline = x
        self.j = j
        self.fixed_present = fixed_present
        self.baseline = torch.sigmoid(self.model(numpy2cuda(x)))
        self.baseline_value = baseline_value # if baseline is not zero

    def forward(self, data):
        '''
        forward pass of model, returns predictions.

        args:
            data: list of np arrays (note, this is different __call__)
        return:
            P(y|x) and predictions
        '''
        x = []
        x_len = []
        for tkn_list in data:
            if len(tkn_list[0, :]) < 400:
                tmp = np.concatenate((tkn_list[0, :], np.zeros(400 - len(tkn_list[0, :]), dtype = 'int')), axis = 0)
                tmp_len = len(tkn_list[0, :])
            else:
                tmp = tkn_list[0,:400]
                tmp_len = 400
            x.append(tmp)
            x_len.append(tmp_len)

        x = np.array(x, dtype = 'int')
        x = numpy2cuda(x)
        x = x.long()
        with torch.no_grad():
            output = self.model(x)
            pred = (output >= 0.0)*1
        return tensor2numpy(torch.sigmoid(output)), tensor2numpy(pred)


    def eval_accy(self, data, label):
        '''
        given samples and labels, calculate accuracy

        args:
            data: list of np arrays (note, this is different __call__)
            label: np array of labels
        '''

        _, pred = self.forward(data)
        pred = numpy2cuda(pred)
        label = numpy2cuda(label)

        truth = label >= 0.5
        accy = pred.eq(truth).sum().item()
        return accy / len(label)

    def __call__(self, x, **kwargs):
        if self.baseline is None: raise Exception('Need to first initialize baseline in evaluation function!')

        with torch.no_grad():

            # Interaction Shapley---------------------------------
            if self.j is not None:                               #
                if self.fixed_present:                           #
                    j_vector = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(x.shape[0], -1)
                    x = np_insert(x, j_vector, index = self.j)   #
                else:                                            #
                    j_vector = (np.zeros((x.shape[0], self.x_baseline.shape[1])) + self.baseline_value)[:,self.j]
                    x = np_insert(x, j_vector, index = self.j)   #
            #-----------------------------------------------------

            x = numpy2cuda(x.astype('int'))
            pred = self.model(x)
            pred = torch.sigmoid(pred)
            if self.baseline < 0.5: pred = 1-pred
        return tensor2numpy(pred)

class eval_image():
    def __init__(self, model, binary = True, reshape = False):
        self.model = model
        self.model.eval()

        self.baseline = None
        self.binary = binary
        self.reshape = reshape

    def init_baseline(self, x, **kwargs):
        '''
        set baseline prediction for original non-perturbed x value
        args:
            x: single sample. numpy array. 1 x c x h x w
        '''
        x = numpy2cuda(x)
        _, self.c, self.h, self.w = x.shape


        if self.binary:
            self.baseline = torch.sigmoid(self.model(x))
        else:
            self.baseline = self.model(x).argmax(dim = 1)

    def forward(self, x):
        '''
        forward pass of model, returns predictions.

        args:
            data: list of np arrays (note, this is different __call__)
        return:
            P(y|x) and predictions
        '''
        x = numpy2cuda(x)

        with torch.no_grad():
            output = self.model(x)

            if self.binary:
                pred = output >= 0.0
            else:
                pred = output.max(1, keepdim=True)[1]  # Calculate Predictions
        
        return tensor2numpy(output), tensor2numpy(pred)

    def eval_accy(self, x, label):
        '''
        given samples and labels, calculate accuracy

        args:
            x: np matrix
            label: np array of labels
        '''

        _, pred = self.forward(x)
        pred = numpy2cuda(pred)
        label = numpy2cuda(label)

        if self.binary:
            truth = label >= 0.5
            accy = pred.eq(truth).sum().item()
        else:
            accy = pred.eq(label.view_as(pred)).sum().item()

        return accy/len(label)

    def __call__(self, x, **kwargs):
        '''
        Note: The input to Shapley Function will be flattened. Therefore, it may be necessary to reshape x prior to a forward pass.
        '''
        if self.baseline is None: raise Exception('Need to first initialize baseline in evaluation function!')

        with torch.no_grad():
            x = numpy2cuda(x)
            x = x.reshape(-1, self.c, self.h, self.w).type(dtype=torch.float32)

            pred = self.model(x)

            if self.reshape:
                pred = tensor2numpy(pred)
                return pred

            if self.binary:
                pred = torch.sigmoid(pred)
                if self.baseline < 0.5: pred = 1-pred
            else:
                pred = torch.exp(-F.cross_entropy(pred, self.baseline.expand(pred.shape[0]), reduction = 'none'))
        return pred.cpu().detach().numpy()


class eval_image_superpixel():
    def __init__(self, model, binary = True):
        self.model = model
        self.model.eval()

        self.baseline = None
        self.binary = binary

    def init_baseline(self, x, num_superpixels, sp_mapping, baseline_value = 0, **kwargs):
        '''
        set baseline prediction for original non-perturbed x value
        args:
            x: single sample. numpy array. 1 x c x h x w
            sp_mapping: superpixel to pixel decoder function
        '''
        x = numpy2cuda(x)
        _, self.c, self.h, self.w = x.shape
        self.x_baseline = x

        # Superpixel mapping
        self.sp_mapping = sp_mapping
        
        # Calculate superpixel map for current sample
        _, self.segment_mask = self.sp_mapping(torch.ones((1, num_superpixels)), x_orig = x)

        if self.binary:
            self.baseline = torch.sigmoid(self.model(x))
        else:
            self.baseline = self.model(x).argmax(dim = 1)
        self.baseline_value = baseline_value # if baseline is not zero

        
    def __call__(self, x, w, **kwargs):
        '''
        args:
            x: superpixel indicator: numpy array
            w: baseline value to set for "null" pixels.
        '''
        if self.baseline is None: raise Exception('Need to first initialize baseline in evaluation function!')

        w = numpy2cuda(w)
        if len(w[0, ...]) == len(x[0, ...]):
            # zero baseline
            w = torch.zeros((x.shape[0], self.c, self.h, self.w))
            w = tensor2cuda(w)
        else:
            # mean baseline
            w = w.reshape(-1, self.c, self.h, self.w)

        with torch.no_grad():
            x = numpy2cuda(x)
            mask, _ = self.sp_mapping(x, x_orig = self.x_baseline, segment_mask = self.segment_mask)
            mask = tensor2cuda(mask)

            x = torch.mul(mask, self.x_baseline) + torch.mul(1-mask, w)

            pred = self.model(x)
            if self.binary:
                pred = torch.sigmoid(pred)
                if self.baseline < 0.5: pred = 1-pred
            else:
                pred = torch.exp(-F.cross_entropy(pred, self.baseline.expand(pred.shape[0]), reduction = 'none'))

        return pred.cpu().detach().numpy()


import xgboost as xgb
from sklearn import metrics

class eval_XGB():
    def __init__(self, model, reshape = False):
        self.model = model
        self.j = None
        self.i = None
        self.reshape = reshape

    def init_baseline(self, x, j = None, i=None, fixed_present = True, baseline_value = 0, **kwargs):
        '''
        set baseline prediction for original non-perturbed x value
        args:
            x: single sample. numpy array. 1 x d
        '''

        _, self.d = x.shape
        self.x_baseline = x
        self.j = j
        self.i = i
        self.fixed_present = fixed_present
        self.baseline = self.model.predict(xgb.DMatrix(x))
        self.baseline_value = baseline_value # if baseline is not zero

    def forward(self, x):
        '''
        forward pass of model, returns predictions.

        args:
            data: list of np arrays (note, this is different __call__)
        return:
            P(y|x) and predictions
        '''

        # Shapley Excess--------------------------------------
        # feature i and j are assumed to be in the same coalition, therefore j is present if i is present
        if self.i is not None:
            j_indicator = (self.x_baseline[:, self.i] == x[:,self.i]).reshape(-1,1)*1 # 1 if j should be present, 0 if j should be absent

            j_present = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(-1, 1)
            j_absent = (np.zeros((x.shape[0], self.x_baseline.shape[1])) + self.baseline_value)[:,self.j].reshape(-1,1)

            j_vector = j_indicator * j_present + (1-j_indicator) *  j_absent
            x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------


        # Interaction Shapley---------------------------------
        if (self.j is not None) and (self.i is None):                               #
            if self.fixed_present:                           #
                j_vector = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(x.shape[0], -1)
                x = np_insert(x, j_vector, index = self.j)   #
            else:                                            #
                j_vector = np.zeros((x.shape[0], 1))         #
                x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------
        pred = self.model.predict(xgb.DMatrix(x))

        
        return pred, (pred > 0.5)*1

    def eval_accy(self, x, label):
        
        _, pred = self.forward(x)
        return metrics.accuracy_score(label, pred)

    def __call__(self, x, **kwargs):
        '''
        Note: The input to Shapley Function will be flattened. Therefore, it may be necessary to reshape x prior to a forward pass.
        '''

        output, _ = self.forward(x)

        if self.reshape:
            output = output.reshape(-1, 1)
            output = np.concatenate((np.ones_like(output) - output, output), axis = 1)
            return output          
        if self.baseline < 0.5: output = 1-output
        return output
'''
class eval_XGB_cox():
    def __init__(self, model):
        self.model = model

    def init_baseline(self, x, **kwargs):

        _, self.d = x.shape


    def forward(self, x):

        pred = self.model.predict(xgb.DMatrix(x), ntree_limit = 5000)

        
        return None, pred

    def eval_accy(self, x, label):
        # adapted from https://github.com/slundberg/shap
        
        _, pred = self.forward(x)
        total = 0
        matches = 0
        for i in range(len(label)):
            for j in range(len(label)):
                if label[j] > 0 and abs(label[i]) > label[j]:
                    total += 1
                    if pred[j] > pred[i]:
                        matches += 1
        return matches/total

    def __call__(self, x, **kwargs):
        #x = pd.DataFrame(x)
        pred = self.model.predict(xgb.DMatrix(x), ntree_limit = 5000)

        return pred
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

class eval_RF_binary():
    def __init__(self, model, binary = True, reshape = False):
        self.model = model

        self.baseline = None
        self.j = None
        self.i = None
        self.binary = binary
        self.reshape = reshape

    def init_baseline(self, x, j = None, i = None, fixed_present = True, baseline_value = 0, **kwargs):
        '''
        set baseline prediction for original non-perturbed x value
        args:
            x: single sample. numpy array. 1 x d
        '''
        _, self.d = x.shape
        self.x_baseline = x
        self.j = j
        self.i = i
        self.fixed_present = fixed_present
        self.baseline = self.model.predict_proba(x)[:,1]
        self.baseline_value = baseline_value # if baseline is not zero

    def forward(self, x, logits = False):
        '''
        forward pass of model, returns predictions.

        args:
            data: list of np arrays (note, this is different __call__)
        return:
            P(y|x) and predictions
        '''

        # Shapley Excess--------------------------------------
        # feature i and j are assumed to be in the same coalition, therefore j is present if i is present
        if self.i is not None:
            j_indicator = (self.x_baseline[:, self.i] == x[:,self.i]).reshape(-1,1)*1 # 1 if j should be present, 0 if j should be absent

            j_present = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(-1, 1)
            j_absent = (np.zeros((x.shape[0], self.x_baseline.shape[1])) + self.baseline_value)[:,self.j].reshape(-1,1)

            j_vector = j_indicator * j_present + (1-j_indicator) *  j_absent
            x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------


        # Interaction Shapley---------------------------------
        if (self.j is not None) and (self.i is None):                               #
            if self.fixed_present:                           #
                j_vector = self.x_baseline[:, self.j].repeat(x.shape[0], axis = 0).reshape(x.shape[0], -1)
                x = np_insert(x, j_vector, index = self.j)   #
            else:                                            #
                j_vector = np.zeros((x.shape[0], 1))         #
                x = np_insert(x, j_vector, index = self.j)   #
        #-----------------------------------------------------
        
        x = tensor2numpy(x)
        output = self.model.predict_proba(x)[:, 1]
        if logits: output = np.log(np.minimum(np.maximum(output, 0.0001), 0.9999)) # logits
        return output, self.model.predict(x)

    def eval_accy(self, x, label):
        '''
        given samples and labels, calculate accuracy

        args:
            x: np matrix
            label: np array of labels
        '''
        _, pred = self.forward(x)
        return metrics.accuracy_score(label, pred)

    def __call__(self, x,**kwargs):
        '''
        Note: The input to Shapley Function will be flattened. Therefore, it may be necessary to reshape x prior to a forward pass.
        '''
        if self.baseline is None: raise Exception('Need to first initialize baseline in evaluation function!')

        output, _ = self.forward(x)
        if self.reshape:
            output = output.reshape(-1, 1)
            output = np.concatenate((np.ones_like(output) - output, output), axis = 1)
            return output

        if self.baseline < 0.5: output = 1-output
        return output
