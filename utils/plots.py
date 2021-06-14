import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import sem
import pandas as pd
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_default_dtype(torch.double)
if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    
def get_loss_mcc(f1):
    '''
    Retrieve data (training loss, MCC sources training batch, MCC sources training set, validation loss, validation MCC sources, mixing matrix MCC) from checkpoints.
    : param f1: (str) path to the checkpoint files. It should be the same data and learning seeds, but varying epochs. e.g. 417_learn-seed_1_*.pth'.
    :return:  each item (e.g. training loss) as an array containing e.g. the training loss for each epoch.
    '''
    losses1 = []
    mcc1 = []
    all_mcc1 = []
    val_losses1 = []
    val_mcc1 = []
    mix_mcc1 = []

    for ckpt in f1:
        loss = torch.load(ckpt, map_location=device)['loss']
        mcc = torch.load(ckpt, map_location=device)['perf']
        all_mcc = torch.load(ckpt, map_location=device)['mcc_dataset']
        val_loss = torch.load(ckpt, map_location=device)['val_loss']
        val_mcc = torch.load(ckpt, map_location=device)['val_mcc']
        mix_mcc = torch.load(ckpt, map_location=device)['mix_mcc']

        losses1.append(loss)
        mcc1.append(mcc)
        all_mcc1.append(all_mcc)
        val_losses1.append(val_loss)
        val_mcc1.append(val_mcc)
        mix_mcc1.append(mix_mcc)
        
    return np.asarray(losses1), np.asarray(mcc1), np.asarray(all_mcc1), np.asarray(val_losses1), np.asarray(val_mcc1), np.asarray(mix_mcc1)

def get_joint(X, n_per_seg=2000, n_seg=40, verbose=False):
    '''
    Compute the joint observation conditioned on the segment p(x_1, x_2 | u) in the binary case:
    - p(x_1=0, x_2=0)
    - p(x_1=0, x_2=1)
    - p(x_1=1, x_2=0)
    - p(x_1=1, x_2=1)
    where x_1 and x_2 are the observations (obtained from each component).
    
    X: (np.ndarray, shape=(80000,2)) Observations array
    n_per_seg: (int) number of points per segment
    n_seg: (int) number of segments
    verbose: (bool) if True, print all the probabilities for each segment.
    '''
    p00s = []
    p11s = []
    p01s = []
    p10s = []

    for i in range(0,n_seg):
        inf = i*n_per_seg
        sup = inf + n_per_seg
        segment = X[inf:sup]

        p00 = len(np.where((segment[:,0] == 0) & (segment[:,1] == 0))[0])/len(segment[:,0])
        p11 = len(np.where((segment[:,0] == 1) & (segment[:,1] == 1))[0])/len(segment[:,0])
        p01 = len(np.where((segment[:,0] == 0) & (segment[:,1] == 1))[0])/len(segment[:,0])
        p10 = len(np.where((segment[:,0] == 1) & (segment[:,1] == 0))[0])/len(segment[:,0])

        if verbose:
            print("Segment ", i)

            print("p(x_1=0, x_2=0) =", p00)
            print("p(x_1=1, x_2=1) =", p11)
            print("p(x_1=0, x_2=1) =", p01)
            print("p(x_1=1, x_2=0) =", p10)

            print("---")

        p00s.append(p00)
        p11s.append(p11)
        p01s.append(p01)
        p10s.append(p10)
    return p00s, p11s, p01s, p10s

def obs_mean(X, n_per_seg=2000, n_seg=40, verbose=False):
    '''
    Compute the mean, variance, and standard error of the binary observations X=[x_1, x_2] per segment u_i.
    Since the observations are 0 or 1, a mean of 0.5 would indicate as many 0s as 1s per segment.
    
    X: (np.ndarray, shape=(80000,2)) Observations array
    n_per_seg: (int) number of points per segment
    n_seg: (int) number of segments
    verbose: (bool) if True, print all the probabilities for each segment.
    '''
    x_means_seg = []
    x_var_seg = []
    x_errors_seg = []

    for i in range(0,n_seg):
        inf = i*n_per_seg
        sup = inf + n_per_seg
        segment = X[inf:sup]
        
        mean = np.mean(segment, axis=0)
        var = np.var(segment, axis=0)
        error = sem(segment)
        
        if verbose:
            print('** Segment ', i)
            print('Mean ', mean)
            print('Variance ', var)
            print('Std error ', error)
            print('-------------')
        
        x_means_seg.append(mean)
        x_var_seg.append(var)
        x_errors_seg.append(error)
        
    x_means_seg = np.asarray(x_means_seg)
    x_var_seg = np.asarray(x_var_seg)
    x_errors_seg = np.asarray(x_errors_seg)

    return x_means_seg, x_var_seg, x_var_seg


def get_vector(seg, n_product):
    '''
    Compute the joint probability of observations given a segment u: p(x_1, ..., x_d|u).
    @param (numpy.ndarray) seg: observations of a given segment. Shape nps X d_sources.
    @param (numpy.ndarray) n_product: n-ary Cartesian power {0,1}^n. e.g. [[0 0 0 0], ..., [1 1 1 1] ] for n=4.
    
    @return (list) joint probability for each element of the n-Cartesian power (each combination).
    '''
    vec = []
    for j in range(len(n_product)):
        temp = [ (seg[i] == n_product[j]).all() for i in range(len(seg)) ]
        temp = np.asarray(temp)
        freq = len(np.where(temp == True)[0])
        p = freq / len(seg)
        vec.append(p)    
    return vec


def get_joint_u(data, nps, n_seg):
    '''
    Compute the joint probability of observations given a segment u: p(x_1, ..., x_d|u), for all the segments u.
    @param (npz) data: data array, should contain observations in data['x'].
    @param (int) nps: number of points per segment.
    @param (int) n_seg: number of segments.
    @return (np.ndarray) joint probabilities, shape (n_seg, 2^d).
    '''
    d = data['x'].shape[-1]
    
    binary_set = set([0,1])
    # n-ary Cartesian power
    n_product = [p for p in itertools.product(binary_set, repeat=d)]
    n_product = np.asarray(n_product)

    all_vec_u = []
    for u in range(0,n_seg):
        idx = u*nps
        seg_u = data['x'][idx:idx+nps]
        vec_u = get_vector(seg_u, n_product)
        all_vec_u.append(vec_u)

    return np.asarray(all_vec_u)


def tol_mean_sem(all_full_mcc, max_epoch):
    '''
    Tolerant mean and standard error of the mean, for computing the statistics when the arrays have different sizes.
    @param all_full_mcc: (list) score of interest. Each element is a 1D array.
    @param max_epoch: (int) maximum length across all arrays in the list.
    @return: mean, standard error arrays across all the learning seeds (elements of the list).
    '''
    arr = np.ma.empty((max_epoch, len(all_full_mcc)))
    arr.mask = True
    for i in range(len(all_full_mcc)): # loop over each learning seed
        arr[:all_full_mcc[i].shape[0], i] = all_full_mcc[i]
    mean = arr.mean(axis=1)
    error = arr.std(axis=1)/np.sqrt(arr.count(axis=1))
    return mean, error


def get_best_seed(data_list, mode='loss'):
    '''
    Get the learning seed that gives the best desired value (the lowest loss or the highest MCS).
    @param data_list: (list) list of strings containing all the data file names (csv) of learning seeds (for each data seed).
    Each file had a different length because each learning seed trains for a different number of epochs.
    @param mode: (str) 'loss' or 'mcs'.
    @return: (int) index of the best seed.
    '''
    data_mix_mcs = [] # mixing matrix MCS
    sources_mcc = [] # MCC Sources full training set
    train_loss = [] # Training batch loss (ELBO)
    epoch_seed = []
    
    # loop over learning seeds
    for log in data_list:
        df = pd.read_csv(log)
        
        train_loss.append(df['Loss'].values[-1])
        data_mix_mcs.append(df['MCS'].values[-1])
        sources_mcc.append(df['MCC_batch'].values[-1])        
        
    data_mix_mcs = np.asarray(data_mix_mcs)
    sources_mcc = np.asarray(sources_mcc)
    train_loss = np.asarray(train_loss)
    
    if mode=='loss':
        idx = np.argmin(train_loss)
    elif mode=='mcs':
        idx = np.argmax(data_mix_mcs)
    return idx