from comet_ml import Experiment
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm
import os
import sys
import pickle

from data import SyntheticDataset
from metrics import mean_corr_coef as mcc
# from metrics import max_perm_mcs_col as mcs_p
from models import cleanIVAE, cleanVAE, Discriminator, permute_dims, simple_IVAE, DiscreteIVAE, discrete_simple_IVAE, u_simple_IVAE

from utils import Logger, checkpoint, full_checkpoint, continuous, disc_plots, plot_all, _disc_plots, inference_plots, gradient_plots, plot_individual
from torch.utils.tensorboard import SummaryWriter

import wandb

from csv import writer

from sklearn.decomposition import FastICA

torch.set_default_dtype(torch.double)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    
    
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
        
    
def runner(args, config):
#     st = time.time() # starting time
    print('Executing script on: {}\n'.format(config.device))

    factor = config.gamma > 0

    print('Data seed: ', str(args.s))
    
#     print('Fixing?')
#     print('fix_prior_mean', args.fix_prior_mean)
#     print('fix_logl', args.fix_logl)

    # checkpoint frequency
    check_freq = config.check_freq

    # Training set
    dset = SyntheticDataset(args.data_path, config.nps, config.ns, config.dl, config.dd, config.nl, args.s, config.p, config.act, uncentered=config.uncentered, noisy=config.noisy, double=factor, one_hot_labels=config.one_hot, simple_mixing=config.simple_mixing, which='train', discrete=config.discrete, identity=config.identity, m_bounds=np.array([-args.m,args.m]), same_var=config.same_var, norm_A_data=config.norm_A_data, norm_logl=config.norm_logl_data, norm_prior_mean=config.norm_mean_data, std_bounds=np.array([config.std_lower, config.std_upper]), diag=config.diag, percentile=config.percentile)
#     , cond_thresh=config.cond_th
    d_data, d_latent, d_aux = dset.get_dims()
    
    # print true mixing matrix
    if ((config.dd < 11) and (config.dl < 11)):
        print('-------------------')
        print('True mixing matrix')
        print(dset.A_mix)
        print('-------------------')

        # print true mixing matrix condition number
        print('Condition number')
        print(np.linalg.cond(dset.A_mix))
        print('-------------------')

    loader_params = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}
    bs = config.nps*config.ns
    data_loader = DataLoader(dset, batch_size=bs, shuffle=False, drop_last=True, **loader_params)

    # Validation set
    val_dset = SyntheticDataset(args.data_path, config.nps, config.ns, config.dl, config.dd, config.nl, args.s, config.p, config.act, uncentered=config.uncentered, noisy=config.noisy, double=factor, one_hot_labels=config.one_hot, simple_mixing=config.simple_mixing, which='val', discrete=config.discrete, identity=config.identity, m_bounds=np.array([-args.m,args.m]), same_var=config.same_var, norm_A_data=config.norm_A_data, norm_logl=config.norm_logl_data, norm_prior_mean=config.norm_mean_data, std_bounds=np.array([config.std_lower, config.std_upper]), diag=config.diag, percentile=config.percentile)
#     , cond_thresh=config.cond_thresh
    val_d_data, val_d_latent, val_d_aux = val_dset.get_dims()

    val_data_loader = DataLoader(val_dset, batch_size=bs, shuffle=False, drop_last=True, **loader_params)

    # Test set for plots
    test_dset = SyntheticDataset(args.data_path, config.nps, config.ns, config.dl, config.dd, config.nl, args.s, config.p, config.act, uncentered=config.uncentered, noisy=config.noisy, double=factor, one_hot_labels=config.one_hot, simple_mixing=config.simple_mixing, which='test', discrete=config.discrete, identity=config.identity, m_bounds=np.array([-args.m,args.m]), same_var=config.same_var, norm_A_data=config.norm_A_data, norm_logl=config.norm_logl_data, norm_prior_mean=config.norm_mean_data, std_bounds=np.array([config.std_lower, config.std_upper]), diag=config.diag, percentile=config.percentile)
#     , cond_thresh=config.cond_thresh


    # to do: pass the paths as parameters from main.py
    if config.checkpoint:
        dir_log = args.dir_log
        ckpt_folder = args.ckpt_folder

        logger = Logger(log_dir=dir_log)
        exp_id = logger.exp_id
        
    # save config file in the checkpoint folder
    file_name = ckpt_folder + str(exp_id) + '/config.p'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pickle.dump(config, open(file_name, "wb"))
    
    # Loop over learning seeds
    for seed in range(args.seed, args.seed + args.n_sims):
        print('*****')
        print("Run ID", exp_id)
        print('*****')

        print("Learning seed ", seed)

        # Save performance
        file_path = str(args.run) + '/' + os.path.splitext(args.config)[0] + '_ds_'+str(args.s) + '_ls_' + str(seed) + '_' + str(exp_id) + '.csv'
        if os.path.exists(file_path):
            os.remove(file_path)
        field_names = ['MCS', 'time', 'MCC']
        append_list_as_row(file_path, field_names)
    
        # load all training data
        Xt, Ut, St, At = dset.x.to(config.device), dset.u.to(config.device), dset.s.to(config.device), torch.from_numpy(dset.A_mix).to(config.device)

        X = Xt.cpu().detach().numpy()
        U = Ut.cpu().detach().numpy()
        S = St.cpu().detach().numpy()
        A = At.cpu().detach().numpy()

        # FastICA
        st = time.time() # starting time
        
        transformer = FastICA(n_components=config.dl, random_state=seed, whiten=True, max_iter=1000)
        S_est = transformer.fit_transform(X)

        ttime_s = time.time() - st # final time
        print('\ntotal runtime: {} seconds'.format(ttime_s))

        mcs = mcc(transformer.mixing_, A, method='cos')
        perf = mcc(S_est, S, method='pearson')
        
        row_contents = [mcs, ttime_s, perf]
        append_list_as_row(file_path, row_contents)