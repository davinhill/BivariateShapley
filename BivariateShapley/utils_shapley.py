#========================================================
# pytorch functions

import torch
import numpy as np
import os
import sys

def tensor2numpy(x):
    if type(x) == torch.Tensor:
        x = x.cpu().detach().numpy()
    return x

def list2cuda(list):
    # adapted from https://github.com/MadryLab/robustness
    array = np.array(list)
    return numpy2cuda(array)

def numpy2cuda(array):
    # function borrowed from https://github.com/MadryLab/robustness
    tensor = torch.from_numpy(array)
    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    # function borrowed from https://github.com/MadryLab/robustness
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def load_model(path):
    tmp = os.path.dirname(os.path.abspath(path))
    sys.path.append(tmp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = torch.load(path, map_location=device)

    return model

def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return min(lr)

# IFG
from torch.distributions.uniform import Uniform
def sample_from_gumbel_softmax_parallel(group_weights, tau=0.1):
    batch_size = group_weights.size(0)
    num_features = group_weights.size(1)
    num_groups = group_weights.size(2)
    logits_ = group_weights
    uniform_dist = Uniform(1e-30, 1.0, )
    uniform = uniform_dist.rsample(sample_shape=group_weights.shape)
    gumbel = -torch.log(-torch.log(uniform))
    gumbel = tensor2cuda(gumbel)
    noisy_logits = (gumbel + logits_) / tau
    samples = torch.nn.Softmax(dim=-1)(noisy_logits)

    return samples
#========================================================
# io functions

import pickle
import os

def save_dict(dictionary, path):
    with open(path, 'wb') as handle:
        pickle.dump(dictionary, handle, protocol = pickle.HIGHEST_PROTOCOL)

def load_dict(path):
    with open(path, 'rb') as handle:
        dictionary = pickle.load(handle)
    return dictionary

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def chdir_script():
    '''
    Changes current directory to that of the current python script
    '''
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

def submit_slurm(python_script, job_file, conda_env='a100', partition='gpu',mem=32, time_hrs = -1, n_gpu = 1, exclude_nodes = None, job_name = 'script'):
    '''
    submit batch job to slurm

    args:
        exclude_nodes: list of specific nodes to exclude
    '''
    dname = os.path.dirname(python_script)
    job_out = os.path.join(dname, 'job_out')
    make_dir(job_out)  # create job_out folder

    if partition not in ['gpu', 'short', 'ai-jumpstart']:
        raise ValueError('invalid partition specified')

    # default time limits
    time_default = {
        'gpu': 8,
        'short':24,
        'ai-jumpstart':24
    }
    # max time limits
    time_max = {
        'gpu': 8,
        'short':24,
        'ai-jumpstart':720
    }
    if time_hrs == -1:
        # set to default time limit
        time_hrs = time_default[partition]
    elif time_hrs > time_max[partition]:
        # set to maximum time limit if exceeded
        time_hrs = time_max[partition]
        warnings.warn('time limit set to maximum for %s partiton: %s hours' % (partition, str(time_hrs)))
    elif time_hrs < 0:
        raise ValueError('invalid (negative) time specified')

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("\n")
        fh.writelines("#SBATCH --job-name=%s\n" % (job_name))
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --tasks-per-node=1\n")
        fh.writelines("#SBATCH --cpus-per-task=1\n")
        fh.writelines("#SBATCH --mem=%sGb \n" % str(mem))
        fh.writelines("#SBATCH --output=" + job_out + "/%j.out\n")
        fh.writelines("#SBATCH --error=" + job_out + "/%j.err\n")
        fh.writelines("#SBATCH --partition=%s\n" % (partition))
        fh.writelines("#SBATCH --time=%s:00:00\n" % (str(time_hrs)))

        # exclude specific nodes
        if exclude_nodes is not None:
            exclude_str = ','.join(exclude_nodes)
            fh.writelines("#SBATCH --exclude=%s\n" % (exclude_str))

        # specify gpu
        if partition == 'gpu':
            fh.writelines("#SBATCH --gres=gpu:v100-sxm2:1\n")
        elif partition == 'ai-jumpstart':
            fh.writelines("#SBATCH --gres=gpu:a100:%s\n" % (str(n_gpu)))

        fh.writelines("\n")
        fh.writelines("module load anaconda3/3.7\n")
        fh.writelines("source activate %s\n" % conda_env)
        fh.writelines("python -u %s" % python_script)
    os.system("sbatch %s" %job_file)




#========================================================
# np functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return x * (x>0)

def softplus(x):
    return np.log(1+np.exp(x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def np_insert(matrix, vector, index):
    '''
    insert vector into matrix (as column) at index
    '''
    matA = matrix[:, :index]
    matB = matrix[:, index:]
    return np.concatenate((matA, vector, matB), axis = 1)

def np_collapse(matrix, index):
    '''
    remove column from matrix
    '''
    matA = matrix[:, :index]
    matB = matrix[:, index+1:]
    return np.concatenate((matA, matB), axis = 1)
#========================================================
# shapley functions



def reshape_uvw(matrix):
    '''
    reshapes a 2d numpy adjacency matrix to a matrix with each edge as a separate row

    input:
        matrix: 2d numpy matrix that represents an adjacency matrix

    return:
        numpy matrix with 3 columns:
        u = source node
        v = target node
        w = edge weight
    '''
    u, v = matrix.shape
    output_matrix = []
    for i in range(u):
        for j in range(v):
            if i != j:
                output_matrix.append([int(i), int(j), matrix[i,j]])
    
    return np.asarray(output_matrix)

def calc_PR_scores(g_adj, personalize = False, shapley_values = None, dmp = 0.85, abs_method = 'softplus'):

    g_adj+= 1e-70  # ensure every node is connected

    if abs_method == 'none':
        g_adj = g_adj
    elif abs_method == 'sigmoid':
        g_adj = sigmoid(g_adj)
    elif abs_method == 'softplus':
        g_adj = softplus(g_adj)
    elif abs_method == 'relu':
        g_adj = relu(g_adj)
    elif abs_method == 'abs':
        g_adj = np.abs(g_adj)

    # sort pagerank / shapley values
    if personalize:

        # ensure at least one positive personalization weight
        if (shapley_values > 0).sum() == 0:
            shapley_values_ = shapley_values - shapley_values.max() + 1e-70
        else:
            shapley_values_ = shapley_values

        node_list, score_list = Phi_PageRank(g_adj, dmp = dmp, shapley_values =shapley_values_)
    else:
        node_list, score_list = Phi_PageRank(g_adj, dmp = dmp)

    return score_list[0]

def rename_nodes(G, node_labels):
    '''
    apply node labels to networkx graph

    args:
        G: NetworkX graph object
        node_labels: list of node labels
    
    return:
        graph
    '''
    node_mapping = dict(zip(np.arange(len(node_labels)).tolist(), node_labels))
    return nx.relabel_nodes(G, node_mapping)


def plot_graph(A, node_labels, directed = True, save_path = None, label_edge_weights = False, color_edges = False, edge_width = 4):
    '''
    plot an adjacency matrix

    input:
        A: adjacentry matrix (numpy matrix)
        node_labels: list of node labels (as string)
        directed: whether the graph is directed (boolean)
        save_path: where the plot should be saved (string)

    return:
        saves the plotted adjacency matrix
    '''

    import networkx as nx

    # directed or undirected graph
    if directed:
        G = nx.from_numpy_matrix(np.matrix(A), create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_matrix(np.matrix(A))

    # apply node labels
    mapping = dict(zip(G,node_labels))
    G = nx.relabel.relabel_nodes(G, mapping)

    # apply edge labels
    if label_edge_weights:
        nx.set_edge_attributes(G, {(e[0], e[1]): {'label': e[2]['weight']} for e in G.edges(data=True)})

    # modify edge width
    width = edge_width
    nx.set_edge_attributes(G, {(e[0], e[1]): {'penwidth': scale_edge_weights(e[2]['weight'], A.max(), width)} for e in G.edges(data=True)})
    
    # color negative edges red
    if color_edges:
        nx.set_edge_attributes(G, {(e[0], e[1]): {'color': 'black'} for e in G.edges(data=True) if e[2]['weight']>0})
        nx.set_edge_attributes(G, {(e[0], e[1]): {'color': 'red'} for e in G.edges(data=True) if e[2]['weight']<0})


    # draw graph
    D = nx.drawing.nx_agraph.to_agraph(G)
    D.layout('dot')
    D.draw(save_path)


def scale_edge_weights(weight, max_weight, max_width = 4):
    return abs(weight / max_weight) * max_width


def matrix_expansion(unary, phi_plus, zero_threshold):
    ''' 
    Calculate phi_minus, phi_redundant, and phi_complement matrices from a phi_plus matrix

    args:
        unary: unary shapley values (np array)
        phi_plus: phi_plus matrix
        zero_threshold: threshold for pairwise shapley values to considered 'zero'

    return:
        phi_minus matrix
        phi_redundant matrix (edge weight of 1 for redundant edges, otherwise 0)
        phi_complement matrix (reciprocal of phi_plus edge weights, after converting 0 to zero_threshold)
    '''

    phi_minus = unary.reshape(-1, 1) - phi_plus
    np.fill_diagonal(phi_minus, 0)

    zero_values_pos = 1 * (phi_plus <= zero_threshold) * (phi_plus >= 0) 
    zero_values_neg = 1 * (phi_plus >= -zero_threshold) * (phi_plus < 0) 

    phi_redundant = zero_values_pos + zero_values_neg
    np.fill_diagonal(phi_redundant, 0)

    # project any value within zero_threshold to be the zero_threshold (to avoid divide by zero)
    phi_complement = phi_plus.copy()
    phi_complement = np.multiply(phi_complement, 1-zero_values_pos) + zero_values_pos * zero_threshold
    phi_complement = np.multiply(phi_complement, 1-zero_values_neg) - zero_values_neg * zero_threshold

    phi_complement = np.power(phi_complement, -1)
    np.fill_diagonal(phi_complement, 0)

    #phi_prop = np.tile(unary.reshape(-1, 1), (1, len(unary)))
    #phi_prop = np.divide(phi_plus, phi_prop, out=np.zeros_like(phi_plus), where=phi_prop != 0)

    return phi_minus, phi_redundant, phi_complement

def get_phi_redundant(phi_plus, zero_threshold, transpose = False):
    '''
    return phi_redundant matrix (with redundancy as sinks)

    (phi+)^T <= threshold
    '''
    if transpose:
        phi_redundant = np.transpose((np.abs(phi_plus) <= zero_threshold)*1)
    else:
        phi_redundant = (np.abs(phi_plus) <= zero_threshold)*1
    np.fill_diagonal(phi_redundant, 0)
    return phi_redundant

def g2h(g_matrix, zero_threshold):
    '''
    convert G-graph adjacency matrix to the H-graph adjacency matrix
    '''

    return get_phi_redundant(phi_plus = g_matrix, zero_threshold = zero_threshold)

def get_phi_ratio(shapley_values, phi_plus, zero_threshold):
    '''
    (phi+ / phi)^T <= threshold
    '''
    shapley_values = np.tile(shapley_values.reshape(-1, 1), (1, len(shapley_values)))
    matrix = np.divide(phi_plus, shapley_values, out=np.zeros_like(phi_plus), where=shapley_values != 0)
    matrix = np.transpose((np.abs(matrix) <= zero_threshold)*1)
    np.fill_diagonal(matrix, 0)
    return matrix

def normalize_phi(shapley_values, phi_plus):
    '''
    phi+ / phi
    '''
    shapley_values = np.tile(shapley_values.reshape(-1, 1), (1, len(shapley_values)))
    matrix = np.divide(phi_plus, shapley_values, out=np.zeros_like(phi_plus), where=shapley_values != 0)
    np.fill_diagonal(matrix, 0)
    return matrix


def find_redundancy(shapley_values, phi_plus, zero_threshold, asym_threshold, shapley_threshold = None):
    '''
    identifies redundancies in phi_plus matrix

    args:
        phi_plus: phi_plus matrix (n x n)
    
    return:
        MR: binary mask (lower triangle) where 1 indicates Mutual Redundancy
        AR: binary mask (lower triangle) where 1 indicates Asymmetric Redundancy
        ppu_zero: binary mask for upper triangle, where 1 indicates zero value
        ppl_zero: binary mask for lower triangle, where 1 indicates zero value
    '''
    if shapley_threshold is not None:
        shapley_mask = np.abs(shapley_values.reshape(-1))>=shapley_threshold
        ind = np.where(shapley_mask == True)[0]
        phi_plus = phi_plus[shapley_mask, :]
        phi_plus = phi_plus[:, shapley_mask]
        shapley_values = shapley_values[shapley_mask]
    else:
        ind = np.arange(len(shapley_values))

    # separate upper / lower phi plus matrix
    ppu = np.triu(phi_plus).transpose()
    ppl = np.tril(phi_plus)

    # calculate mutual redundancy
    ppu_zero = (np.abs(ppu) <= zero_threshold)*1
    ppl_zero = (np.abs(ppl) <= zero_threshold)*1
    MR = np.tril(np.multiply(ppu_zero, ppl_zero))
    np.fill_diagonal(MR, 0)

    # calculate Asymmetric Redundancy
    pp_diff = ppu - ppl  # difference between upper and lower triangle
    pp_asym = (np.abs(pp_diff) >= asym_threshold)*1  # asymmetric features
    pp_zero = (ppu_zero + ppl_zero == 1)*1  # either i or j is zero (but not both)
    AR = np.multiply(pp_asym, pp_zero) # features that are asymmetric AND (i = 0 OR j = zero)
    
    return MR, AR, ppu_zero, ppl_zero, shapley_values, ind




def top_R(x, shapley_values, phi_plus, zero_threshold = 0.01, asym_threshold = 0.2, shapley_threshold = None, db = {}):
    '''
    find top redundancies for a given phi_plus matrix

    '''


    if not bool(db): # if db is empty
        db['counter_MR'] = 0
        db['counter_AR'] = 0
        db['counter_masked_AR'] = 0
        db['counter_masked_AR_flip'] = 0
        db['n_masked_AR_list'] = [] # number of features masked due to AR for each sample
        db['AR_shapley_distn'] = [] # always reset to empty

    MR, AR, ppu_zero, ppl_zero, shapley_values, ind = find_redundancy(shapley_values, phi_plus, zero_threshold, asym_threshold, shapley_threshold = shapley_threshold)

    ######################
    # calculate mutual redundancy
    rows, cols = np.where(MR == 1)

    x_MR = x.clone().cpu().detach().numpy()
    MR_list = np.zeros((len(rows), 2), dtype = int)
    if len(rows) != 0:
        '''
        # add top MR to list
        for i in range(len(rows)):
        '''
        
        for idx, (i, j) in enumerate(zip(rows, cols)):
            MR_list[idx, :] = np.sort(np.array((x_MR[0, i], x_MR[0, j]), dtype = int)).reshape(-1)

        for idx, (i, j) in enumerate(zip(rows, cols)):
            # randomly drop one of the redundant features
            b = np.random.binomial(n=1, p=0.5)
            x_MR[:, i] = int(x_MR[:, i] * b)
            x_MR[:, j] = int(x_MR[:, j] * (1-b))

        db['counter_MR'] += 1

    ###################
    # calculate Asymmetric Redundancy
    rows, cols = np.where(AR == 1)

    x_AR = x.clone().cpu().detach().numpy()
    x_AR_flip = x.clone().cpu().detach().numpy()
    AR_list = np.zeros((len(rows), 2), dtype = int) # the redundant feature is first
    tmp_list = []
    if len(rows) != 0:
        # record the redundant feature
        for idx, (i, j) in enumerate(zip(rows, cols)):
            if ppu_zero[i, j] == 1: # if upper triangle of ppu is zero: given i, j is redundant
                AR_list[idx, :] = np.array((x_AR[0, ind[j]], x_AR[0, ind[i]]), dtype = int)
                tmp_list.append(np.abs(shapley_values[j]))
            elif ppl_zero[i, j] == 1: # if lower triangle of ppu is zero: given j, i is redundant
                AR_list[idx, :] = np.array((x_AR[0, ind[i]], x_AR[0, ind[j]]), dtype = int)
                tmp_list.append(np.abs(shapley_values[i]))

        # drop the redundant feature
        for idx, (i, j) in enumerate(zip(rows, cols)):
            if ppu_zero[i, j] == 1: # if upper triangle of ppu is zero: given i, j is redundant
                x_AR[:, ind[j]] = 0
                x_AR_flip[:, ind[i]] = 0
            elif ppl_zero[i, j] == 1: # if lower triangle of ppu is zero: given j, i is redundant
                x_AR[:, ind[i]] = 0
                x_AR_flip[:, ind[j]] = 0

        # only keep unique masked values
        _, u_ind = np.unique(AR_list[:, 0], return_index=True)
        tmp_list = np.array(tmp_list)[u_ind]

        db['counter_AR'] += 1
        db['n_masked_AR_list'].append(len(np.unique(AR_list[:, 0])))
        db['counter_masked_AR'] += len(np.unique(AR_list[:, 0]))
        db['counter_masked_AR_flip'] += len(np.unique(AR_list[:, 1]))
        db['AR_shapley_distn'].append(tmp_list)
    else:
        db['n_masked_AR_list'].append(0)
        db['AR_shapley_distn'].append([])

    x_MR = x_MR[0, :].tolist()
    x_AR = x_AR[0, :].tolist()
    x_AR_flip = x_AR_flip[0, :].tolist()
    return x_MR, x_AR, x_AR_flip, db, MR_list, AR_list


#========================================================
# transformer functions
from transformers import BertTokenizerFast

def encode(batch, max_length = 400, tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased'), add_special_tokens = False):
    return tokenizer(batch, padding="longest", truncation=True, max_length=max_length, add_special_tokens = add_special_tokens)['input_ids']

def decode(batch, single = False, tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased'), skip_special_tokens = False):
    if single:
        output = []
        output.append(tokenizer.convert_tokens_to_string(tokenizer.batch_decode(batch, skip_special_tokens)))
    else:
        output = tokenizer.batch_decode(batch, skip_special_tokens)
    
    return output

def decode_tkn(batch, tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased'), skip_special_tokens = False):
    output = tokenizer.batch_decode(batch.reshape(-1), skip_special_tokens)
    return output

#========================================================
# nlp functions


def string_to_tkn(caption, stopwords= False):
    import nltk
    nltk.download('punkt')

    if stopwords:
        stop_list = [
            'a',
            'an',
            'and',
            'are',
            'as',
            'at',
            'be',
            'by',
            'for',
            'from',
            'has',
            'he',
            'in',
            'is',
            'it',
            'its',
            'of',
            'on',
            'that',
            'the',
            'to',
            'was',
            'were',
            'will',
            'with'
        ]
    else:
        stop_list = []

    # tokenize caption
    caption_tkn = nltk.word_tokenize(caption)
    caption_tkn = [w.lower() for w in caption_tkn if not w in stop_list]

    # insert start / end tokens
    caption_tkn.insert(0, '<S>')
    caption_tkn.append('</S>')

    return caption_tkn


def caption_to_id(caption, dictionary, vocab_size, max_cap_len):
# ================================
# Input a single caption (string), add start/end/unknown tokens, then convert to IDs
# Note: The dictionary must be first truncated to the vocab size
# ================================

    import nltk
    nltk.download('punkt')

    # tokenize caption
    caption_tkn = nltk.word_tokenize(caption)
    caption_tkn = [w.lower() for w in caption_tkn]

    # insert start / end tokens
    caption_tkn.insert(0, '<S>')
    caption_tkn.append('</S>')


    # initialize
    caption_tknID = [0] * (max_cap_len)

    # insert unknown token
    for i, tkn in enumerate(caption_tkn):
        
        # if caption is longer than max caption length, break.
        # add 3 for the start/stop token and period.
        if i == (max_cap_len):
            break

        # words not in dictionary
        if (tkn not in dictionary):
            tkn = 'UNK'

        # lookup tokenID
        caption_tknID[i] = dictionary.get(tkn)

        # if word is not in dictionary
        if caption_tknID[i] is None:
            caption_tknID[i] = 1

        # truncate dictionary based on vocab_size        
        if caption_tknID[i] >= vocab_size:
            caption_tknID[i] = 1

        # Special Tokens:
        # 0 <p>
        # 1 <UNK>
        # 2 <S>
        # 3 </S>

    # convert word tokens to id
    return (caption_tknID)


def id_to_word(tkn_list, conversion_array):
    '''
    Convert ID to Words
    tkn_list should be a single list of tknIDs. Returns a list of words.
    '''

    tkn_list = tkn_list.cpu().detach().numpy()
    return [conversion_array[tkn] for tkn in tkn_list]

def pad_x(x, max_len):
    '''
    x is a tensor of dimension 1 x d
    '''
    x_len = x.shape[1]
    if x_len < max_len:
        x = torch.cat((x, torch.zeros((1, 400-x.shape[1]), dtype = int)), dim = 1)
    elif x_len >max_len:
        x = x[:, :max_len]
            
    return x


#========================================================
# CV functions


from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

from torchvision import transforms

def superpixel_to_mask(x_superpixel, x_orig, segment_mask = None):
    '''
    input:
        x_superpixel: binary matrix, n x num_superpixels
        x_orig: image tensor, 1 x c x h x w
        segment_mask: optional to save redundant calculations, Represents a flattened binary mask for each superpixel group

    '''

    x_superpixel = tensor2cuda(x_superpixel).float()

    #Note: SLIC requires image to be in h x w x c
    _, c, h, w = x_orig.shape
    #image = x_orig[0, ...].permute(1, 2, 0).cpu()  # h x w x c
    #mask_out = torch.zeros((x_superpixel.shape[0], h, w, c)) # n x h x w x c
    num_segments = x_superpixel.shape[1]

    if segment_mask is None:

        image = x_orig[0, ...].permute(1, 2, 0).cpu()  # h x w x c
        segments = slic(image, n_segments = num_segments, sigma = 5, start_label = 0)
        segment_mask = []
        for i in range(num_segments):
            segment_mask.append(torch.tensor(segments == i, dtype = torch.float32).unsqueeze(0)) # 1 x h x w
        segment_mask = tensor2cuda(torch.cat(segment_mask, dim = 0)) #  num_superpixels x h x w
        segment_mask = segment_mask.unsqueeze(0).expand(x_superpixel.shape[0], -1, -1, -1) # n x num_superpixels x h x w
        segment_mask = torch.flatten(segment_mask, 2)  # n x num_superpixels x (h*w)

    mask_out = torch.matmul(x_superpixel.unsqueeze(1), segment_mask).squeeze(1)  # n x (h*w)
    mask_out = mask_out.reshape(x_superpixel.shape[0], h, w)  # n x h x w
    mask_out = mask_out.unsqueeze(1).expand(-1, c, -1, -1) # n x c x h x w   copy over channels
    ''' 
    for j in range(x_superpixel.shape[0]):
        for i in range(num_segments):
            if x_superpixel[j, i] == 1:
                segment_mask = torch.tensor(segments == i, dtype = torch.int32).unsqueeze(-1).expand(-1, -1, c)
                mask_out[j, ...] = mask_out[j, ...] + segment_mask
    return mask_out.permute(0, 3, 1, 2) # n x c x h x w
    '''
    return mask_out, segment_mask 

'''
need to split this function into 2 in order to use mean baseline.


'''
def create_mask(ind, x, num_superpixels, sp_transform):
    
    if num_superpixels>0:
        superpixel_mask = torch.zeros((num_superpixels))
        superpixel_mask[ind] = 1
        mask, _ = sp_transform(superpixel_mask.reshape(1, -1), torch.tensor(x)) # n x c x h x w
        mask = mask[0, ...].permute(1, 2, 0).cpu().detach().numpy()
    else:
        mask = torch.zeros(x.shape)
        mask[ind] = 1
        mask = mask[0, ...].permute(1, 2, 0).cpu().detach().numpy()
    return mask

class UnNormalize(transforms.Normalize):
    def __init__(self, x_mean, x_std):
        std_inv = 1/x_std
        mean_inv = -x_mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)


    def __call__(self, x_input):
        
        x = x_input.clone()
        for i in range(x.shape[0]):
            x[i, :] = super().__call__(x[i, :])
        return x



#========================================================
# graph functions


import networkx as nx

try:
    from sknetwork.ranking import PageRank
except:
    import warnings
    warnings.warn('Problem importing Scikit-Network!')



def find_MR(phi_redundant):
    '''
    find mutually redundant groups

    args:
        phi_redundant: directed, unweighted adjacency matrix with redundancy as sinks
    return:
        list of np arrays, where each array is a set of indices representing a MR group
    '''
    G = nx.from_numpy_matrix(phi_redundant , create_using=nx.DiGraph)
    components = list(nx.strongly_connected_components(G))
    components_multi = [c for c in components if len(c)>1]


    MR_groups = [] # list of index arrays
    for c in components_multi:
        MR_groups.append(np.array(list(c), dtype = 'int'))

    return MR_groups



def find_AR(phi_redundant, calc_freq = False):
    '''
    find asymmetrically redundant groups

    args:
        phi_redundant: directed, unweighted adjacency matrix with redundancy as sinks
    return:
        list of np arrays, where each array is a set of indices representing a AR group
    '''
    freq_matrix = np.zeros_like(phi_redundant)  # create a matrix to track frequency of sinks/sources

    pagerank = PageRank()

    G = nx.from_numpy_matrix(phi_redundant , create_using=nx.DiGraph)
    components = list(nx.strongly_connected_components(G))

    # condensed graph
    C = nx.condensation(G, components)

    # get connected subgraphs in condensed graph
    S = [C.subgraph(c).copy() for c in nx.weakly_connected_components(C)]
    xx = list(nx.weakly_connected_components(C))  # Nodes of each subgraph

    # calculate condensed node mapping
    node_mapping = []
    for i in range(len(phi_redundant)):
        node_mapping.append(C.graph['mapping'][i])
    node_mapping = np.array(node_mapping)

    # calculate source nodes (condensed graph)
    c_node_source = []
    c_node_sink = []
    for idx, subgraph in enumerate(S):
        adjacency = nx.adjacency_matrix(subgraph)
        if adjacency.todense().shape == (1,1):
            continue
        scores = pagerank.fit_transform(adjacency)

        #list_nodes.append(np.array(list(xx[idx])))
        #list_scores.append(scores)
        
        # Source Nodes
        source_idx = np.where(scores == scores.min())[0] # index of source nodes
        tmp = np.array(list(xx[idx]))[source_idx]  # source nodes (condensed)
        c_node_source.append(tmp)

        ################
        # calculate source/sink frequency for plots
        if calc_freq and len(tmp) != 0:

            # calculate non-source condensed nodes
            mask = np.ones_like(scores)
            mask[source_idx] = 0
            all_other_idx = np.multiply(np.array(list(xx[idx])), mask).nonzero()[0]

            if len(all_other_idx) != 0:
                for c_node in tmp:

                    # reverse condensation
                    nodes = np.where(node_mapping == c_node)[0]  # nodes in the condensed node
                    ao_nodes = np.where(node_mapping == all_other_idx)[0]

                    for node in nodes:
                        freq_matrix[node, ao_nodes] += 1
        ###################

        # Sink Nodes
        if len(np.unique(scores)) > 1:
            sink_idx = np.where(scores == scores.max())[0] # index of sink nodes
            tmp = np.array(list(xx[idx]))[sink_idx]  # sink nodes (condensed)
            c_node_sink.append(tmp)


    ###########################
    # reverse node condensation
    ###########################

    # Source Nodes 
    if len(c_node_source) != 0:
        c_node_source = np.concatenate(c_node_source)

        multi_n = []
        sing_n = []
        for c_node in c_node_source:
            nodes = np.where(node_mapping == c_node)[0]  # nodes in the condensed node
            if len(nodes) >1:
                multi_n.append(nodes)
            else:
                sing_n.append(nodes)

        if len(multi_n) + len(sing_n) == 0:
            all_n = None
        else:
            all_n = np.concatenate(sing_n + multi_n)  # all source nodes (np array)
        source_nodes = [all_n, sing_n, multi_n]
    else:
        source_nodes = [None, [], []]


    # Sink Nodes 
    if len(c_node_sink) !=0 :
        c_node_sink = np.concatenate(c_node_sink)

        multi_n = []
        sing_n = []
        for c_node in c_node_sink:
            nodes = np.where(node_mapping == c_node)[0]  # nodes in the condensed node
            if len(nodes) >1:
                multi_n.append(nodes)
            else:
                sing_n.append(nodes)

        if len(multi_n) + len(sing_n) == 0:
            all_n = None
        else:
            all_n = np.concatenate(sing_n + multi_n)  # all source nodes (np array)
        sink_nodes = [all_n, sing_n, multi_n]
    else:
        sink_nodes = [None, [], []]


    return source_nodes, sink_nodes, freq_matrix


def Phi_PageRank(phi_plus, shapley_values = None, dmp = 0.85):
    '''
    PageRank on Phi_Plus Matrix
    '''

    pagerank = PageRank(damping_factor=dmp)
    G = nx.from_numpy_matrix(phi_plus , create_using=nx.DiGraph)

    for i in range(phi_plus.shape[0]):
        for j in range(phi_plus.shape[0]):
            if phi_plus[i,j] != 0:
                G.add_weighted_edges_from([(i, j, phi_plus[i,j])])

    xx = list(nx.weakly_connected_components(G))

    S = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]

    list_nodes = []
    list_scores = []

    for i in range(len(S)):
        adjacency = nx.adjacency_matrix(S[i])
        if shapley_values is None:
            scores = pagerank.fit_transform(adjacency)
        else:
            scores = pagerank.fit_transform(adjacency, seeds = shapley_values)


        list_nodes.append(np.array(list(xx[i])))
        list_scores.append(scores)

    return list_nodes, list_scores





def absorbing_classes(G):
    '''
    Get all absorbing nodes and subgraphs

    args:
        graph G
    return:
        nodes, subgraphs
    '''
    cpts = [G.subgraph(c).copy() for c in nx.strongly_connected_components(G)]
    cpts_nodes = list(nx.strongly_connected_components(G))
    internal = set()

    for sg in cpts:
        for e in sg.edges():
            internal.add(e)

    # find all the edges that aren't part of the strongly connected components
    # ~ O(E)
    transient_edges = set(G.edges()) - internal

    # find the start of the directed edge leading out from a component
    # ~ O(E)
    transient_srcs = set([ e[0] for e in transient_edges ])
    # yield everything that don't have a vertex in transient_srcs
    list_nodes = []
    list_subgraphs = []
    for idx, sg in enumerate(cpts):
        if transient_srcs - set(sg.nodes()):
            list_nodes.append(cpts_nodes[idx])
            list_subgraphs.append(sg)
    return list_subgraphs, list_nodes

    
def find_reducible_subgraphs(G):
    connected_sg = [G.subgraph(c).copy() for c in nx.weakly_connected_components(G)]
    connected_nodes = list(nx.weakly_connected_components(G))

    reducible_sg = []
    reducible_nodes = []
    for idx, sg in enumerate(connected_sg):
        if not nx.is_strongly_connected(sg):
            reducible_sg.append(sg)
            reducible_nodes.append(connected_nodes[idx])

    return reducible_sg, reducible_nodes

def absorbing_classes_reducible(G):
    
    reducible_sg, _ = find_reducible_subgraphs(G)

    abs_classes = []
    abs_nodes = []
    for sg in reducible_sg:
        tmp_sg, tmp_nodes = absorbing_classes(sg)
        for (i, j) in zip(tmp_sg, tmp_nodes):
            abs_classes.append(i)
            abs_nodes.append(j)
    
    return abs_classes, abs_nodes

def find_AR_sink(phi_redundant):
    G = nx.from_numpy_matrix(phi_redundant , create_using=nx.DiGraph)
    
    _, abs_nodes = absorbing_classes_reducible(G)

    multi_sink = [np.array(list(c)) for c in abs_nodes if len(c)>1]
    sing_sink = [np.array(list(c)) for c in abs_nodes if len(c)==1]
    if len(multi_sink) + len(sing_sink) == 0:
        all_sink = None
    else:
        all_sink = np.concatenate(sing_sink + multi_sink)  # all source nodes (np array)
    return all_sink, sing_sink, multi_sink
