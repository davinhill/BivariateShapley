import torch
import numpy as np
import torch.nn.functional as F
import pickle

def tensor2numpy(x):
    if type(x) == torch.Tensor:
        x = x.numpy()
    return x

def list2cuda(list):
    # adapted from https://github.com/MadryLab/robustness
    array = np.array(list)
    return numpy2cuda(array)

def numpy2cuda(array):
    # adapted from https://github.com/MadryLab/robustness
    tensor = torch.from_numpy(array)
    return tensor2cuda(tensor)

def tensor2cuda(tensor):
    # adapted from https://github.com/MadryLab/robustness
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor

def calc_test_accy_rnn(model, test_loader):
    model.eval()
    with torch.no_grad():
        epoch_accy = 0.0
        epoch_counter = 0
        num_batches = len(test_loader)
        batch_size = test_loader.batch_size
        for idx, batch in enumerate(test_loader):
            data, target = tensor2cuda(batch[0]), tensor2cuda(batch[1])
            data_lens = list2cuda(batch[2])

            pred = model(data, data_lens)

            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_accy += acc
            epoch_counter += batch_size

        epoch_accy /= (num_batches*batch_size)

    return epoch_accy

def calc_test_accy_cnn(model, test_loader):
    model.eval()
    with torch.no_grad():
        epoch_accy = 0.0
        epoch_counter = 0
        num_batches = len(test_loader)
        batch_size = test_loader.batch_size
        for idx, batch in enumerate(test_loader):
            data, target = tensor2cuda(batch[0]), tensor2cuda(batch[1])

            pred = model(data)

            prediction = pred >= 0.0
            truth = target >= 0.5
            acc = prediction.eq(truth).sum().cpu().data.numpy()
            epoch_accy += acc
            epoch_counter += batch_size

        epoch_accy /= (num_batches*batch_size)

    return epoch_accy

def id_to_word(tkn_list, conversion_array):
    '''
    Convert ID to Words
    tkn_list should be a single list of tknIDs. Returns a list of words.
    '''

    tkn_list = tkn_list.cpu().detach().numpy()
    return [conversion_array[tkn] for tkn in tkn_list]

def wordlist_to_string(caption_tkn):
    '''
    convert list of tokenized words to string
    input is a single list of tokenized words
    '''

    # reduce string length if end token is present
    if '<S.' in caption_tkn:
        startindex = 1
    else:
        startindex = 0
    if '</S>' in caption_tkn:

        output = ' '.join(caption_tkn[startindex:caption_tkn.index('</S>')])
    else:
        output = ' '.join(caption_tkn[startindex:])
    
    return output

def calc_test_loss_vae(model, test_loader, anneal_function, ELBO, step, k, x0, max_length):
    model.eval()
    epoch_loss = 0.0
    num_batches = len(test_loader)
    with torch.no_grad():

        for idx, batch in enumerate(test_loader):
            data = tensor2cuda(batch[0])
            data_lens = tensor2cuda(batch[2])

            pred, z, mu, log_var = model(data, data_lens)
            NLL, KL_loss = ELBO(pred,data, data_lens,max_length, KL=True,mu= mu,log_var= log_var)
            KL_weight = kl_anneal_function(anneal_function, step, k, x0)
            loss = (NLL + KL_loss * KL_weight) / data.shape[0]

            epoch_loss += loss.data.item()

        epoch_loss /= num_batches
        return epoch_loss


def calc_test_loss_vaeac(model, test_loader, anneal_function, ELBO, step, k, x0, mask_generator):
    model.eval()
    epoch_loss = 0.0
    num_batches = len(test_loader)
    with torch.no_grad():

        for idx, batch in enumerate(test_loader):
            data = tensor2cuda(batch[0])
            data_lens = tensor2cuda(batch[2])

            mask = tensor2cuda(mask_generator(data, step).type(torch.int32)) # n x max_length
            NLL, KL_loss, prior_reg = model.batch_vlb(data, mask, data_lens)


            KL_weight = kl_anneal_function(anneal_function, step, k, x0) # calculate KL weight
            loss = (NLL + KL_loss * KL_weight - prior_reg).mean()
            epoch_loss += loss.data.item()

        epoch_loss /= num_batches
    model.train()    
    return epoch_loss

def kl_anneal_function(anneal_function, step, k, x0):
    '''
    adapted from https://github.com/timbmg/Sentence-VAE

    '''
    if anneal_function == 'logistic':
        return float(1/(1+np.exp(-k*(step-x0))))
    elif anneal_function == 'linear':
        return min(1, step/x0)


def ELBO(logprob, target, target_lens, max_length, KL = True, mu = None, log_var = None, mask = None):
    '''
    args:
        logprob: from forward pass         n x vocab_size x t (padded)
        target: model input (padded)       n x t (padded)
        target_lens: length of target sequences
        KL:                                If True, return KL loss, else only reconstruction loss
        mu: mu for latent space            n x latent_size
        log_var: log-var for latent space  n x latent_size
    
    return:
        loss

    '''
    batch_size = target.shape[0]

    logprob = logprob[:, :, :-1]  # remove last word in prediction
    target = target[:, 1:]  # remove first word in target    

    # pad target/pred if not max_length
    if target.shape[1] != max_length -1:
        target = torch.cat((target, tensor2cuda(torch.zeros((target.shape[0], max_length -1  - target.shape[1]), dtype = torch.int32))), dim = 1)
    if logprob.shape[2] != max_length -1:
        logprob = torch.cat((logprob, tensor2cuda(torch.zeros((logprob.shape[0], logprob.shape[1],max_length -1  - logprob.shape[2]), dtype = torch.int32))), dim = 2)        

    # reshape target / pred for loss
    target = target.reshape(batch_size * (max_length-1))  # batch_size * (max_length-1)
    logprob = logprob.transpose(1, 2).reshape(batch_size * (max_length-1), -1)  # batch_size * (max_length-1) x vocab_size


    # if training VAEAC and masking out loss
    if mask is not None:
        mask = mask[:, 1:]  # remove first word in mask
        if mask.shape[1] != max_length -1:
            mask = torch.cat((mask, tensor2cuda(torch.zeros((mask.shape[0], max_length -1  - mask.shape[1]), dtype = torch.int32))), dim = 1)
        mask = mask.reshape(batch_size * (max_length-1))  # batch_size * (max_length-1)

        word_mask = torch.mul(target, mask).nonzero(as_tuple = False).reshape(-1)  # filter out unused words when target is shorter than max_length
    else:
        word_mask = target.nonzero(as_tuple = False).reshape(-1)  # filter out unused words when target is shorter than max_length

    NLL = F.nll_loss(logprob[word_mask, :], target[word_mask], reduction = 'sum')

    if KL:
        # Calculate KL Loss
        KL_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return NLL, KL_loss

    else:
        return NLL



class MCARGenerator:
    """
    Adapted from https://github.com/tigvarts/vaeac

    Returned mask is sampled from component-wise independent Bernoulli
    distribution with probability of component to be unobserved p.
    Such mask induces the type of missingness which is called
    in literature "missing completely at random" (MCAR).

    If some value in batch is missed, it automatically becomes unobserved.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, batch, step):
        # step parameter is unused, it's included to make it easier switching between mask functions

        nan_mask = torch.isnan(batch).float()  # missed values
        bernoulli_mask_numpy = np.random.choice(2, size=batch.shape,
                                                p=[1 - self.p, self.p])
        bernoulli_mask = torch.from_numpy(bernoulli_mask_numpy).float()
        bernoulli_mask = tensor2cuda(bernoulli_mask)
        mask = torch.max(bernoulli_mask, nan_mask)  # logical or
        return mask

class MCARGenerator_annealed:
    """
    Similar to MCARGenerator, but annealed
    """
    def __init__(self, max_p, min_p, epochs):
        self.max_p = max_p
        self.min_p = min_p
        self.epochs = epochs

    def __call__(self, batch, step):
        # p starts at max p and anneals linearly to min p
        p = max(0, min(1, (self.epochs - step) / self.epochs)) * (self.max_p-self.min_p) + self.min_p

        nan_mask = torch.isnan(batch).float()  # missed values
        bernoulli_mask_numpy = np.random.choice(2, size=batch.shape,
                                                p=[1 - p, p])
        bernoulli_mask = torch.from_numpy(bernoulli_mask_numpy).float()
        bernoulli_mask = tensor2cuda(bernoulli_mask)
        mask = torch.max(bernoulli_mask, nan_mask)  # logical or
        return mask

def find_redundancy(phi_plus, zero_threshold, asym_threshold):
    '''
    identifies redundancies in phi_plus matrix

    args:
        phi_plus: phi_plus matrix (n x n)
    
    return:
        MR: binary mask (lower triangle) where 1 indicates Mutual Redundancy
        AR: binary mask (lower triangle) where 1 indicates Asymmetric Redundancy
    '''
    # separate upper / lower phi plus matrix
    ppu = np.triu(phi_plus).transpose()
    ppl = np.tril(phi_plus)

    # calculate mutual redundancy
    ppu_zero = np.multiply(ppu >= -zero_threshold, ppu <= zero_threshold)
    ppl_zero = np.multiply(ppl >= -zero_threshold, ppl <= zero_threshold)
    MR = np.tril(np.multiply(ppu_zero, ppl_zero))

    # calculate Asymmetric Redundancy
    pp_diff = ppu - ppl  # difference between upper and lower triangle
    pp_asym = np.abs(pp_diff) >= asym_threshold  # asymmetric features
    pp_zero = ppu_zero + ppl_zero > 0  # either i or j is zero
    AR = np.multiply(pp_asym, pp_zero) # features that are asymmetric AND (i = 0 OR j = zero)
    
    return MR, AR, ppu_zero, ppl_zero
