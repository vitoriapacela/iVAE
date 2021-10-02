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

    if config.tensorboard:
        tensorboard_dir = ckpt_folder + 'tensorboard/ivae/' + str(exp_id) + '/'

    # save config file in the checkpoint folder
    file_name = ckpt_folder + str(exp_id) + '/config.p'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    pickle.dump(config, open(file_name, "wb"))
        
    ## Start training
    for seed in range(args.seed, args.seed + args.n_sims):
        print('*****')
        print("Run ID", exp_id)
        print('*****')
        
        print("Learning seed ", seed)

        if config.wandb:
            run_name = str(exp_id) + "_ls_" + str(seed) + "_ds_" + str(args.s)
            wandb.init(project="ivae", config=config, name=run_name, reinit=True, dir=ckpt_folder+'wandb/')
            wandb.config.update(args)

        if config.comet:
            experiment = Experiment(api_key="vnYqg73uIJr8Efrt6gd9TCJU7", project_name="iVAE", workspace="vitoriapacela")
            experiment.log_parameters(config)
            experiment.set_name(str(exp_id)+"_ls_"+str(seed)+"_ds_"+str(args.s))

        if config.ica:
            if config.discrete:
                model = discrete_simple_IVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim, n_layers=config.n_layers, activation=config.activation, slope=.1, interpret=config.terms, simple_g=config.g, device=config.device, simple_prior=config.simple_prior, simple_logv=config.simple_logv, fix_v=config.fix_v, logvar=config.logvar).to(config.device)
            else:
                if config.uncentered:
                    model = u_simple_IVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim, n_layers=config.n_layers, activation=config.activation, slope=.1, interpret=config.terms, simple_g=config.g, simple_prior = config.simple_prior, simple_logv=config.simple_logv, noise_level=config.noisy).to(config.device)
                else:
                    model = simple_IVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim, n_layers=config.n_layers, activation=config.activation, slope=.1, interpret=config.terms, simple_g=config.g, simple_prior=config.simple_prior, simple_logv=config.simple_logv, noise_level=config.noisy).to(config.device)

            if config.verbose:
                print(model)
            
                # Count number of parameters
                model_parameters = filter(lambda p: p.requires_grad, model.parameters())
                params = sum([np.prod(p.size()) for p in model_parameters])
                print('Number of parameters', params)
            
            if args.set_f:
                # For debugging purposes
                # Fix the mixing model (decoder) weights to the true mixing matrix values
                with torch.no_grad():
                    # The following will assume a square mixing matrix
                    for i in range(d_data):
                        for j in range(d_latent):
                            model.f.fc[0].weight[i,j] = torch.nn.Parameter(torch.tensor(dset.A_mix[i,j]))
                        
                # Freeze such weights during the optimization
                model.f.fc[0].requires_grad_(False)
                
            if args.close_f:
                # For debugging purposes
                # Initialize the mixing model (decoder) weights to the true mixing matrix values
                with torch.no_grad():
                    # The following will assume a square mixing matrix
                    for i in range(d_data):
                        for j in range(d_latent):
                            model.f.fc[0].weight[i,j] = torch.nn.Parameter(torch.tensor(dset.A_mix[i,j])) + args.init_noise*(torch.randn(1)) # noise from init_noise * N(0, 1)                            
                        
                # Do NOT freeze such weights during the optimization
                model.f.fc[0].requires_grad_(True)
                
                
            if args.set_prior:
                # set weights to true prior
                with torch.no_grad():
                    model.prior_mean.fc[0].weight = torch.nn.Parameter(torch.tensor(dset.m.T)) # dset.m are the true means
                    model.logl.fc[0].weight = torch.nn.Parameter(torch.tensor(np.log( (dset.l.T)**2 ))) # dset.l are the true stds, so we convert them to logvar

                     # Freeze such weights during the optimization
                    model.prior_mean.fc[0].requires_grad_(False)
                    model.logl.fc[0].requires_grad_(False)
                    
             # fix only prior mean model
            if args.fix_prior_mean:
                # init weights to true prior
                with torch.no_grad():
                    model.prior_mean.fc[0].weight = torch.nn.Parameter(torch.tensor(dset.m.T)) # dset.m are the true means
                # Freeze such weights during the optimization
                for param in model.prior_mean.parameters():
                    param.requires_grad = False
                    
            # fix only prior var model        
            if args.fix_logl:
                with torch.no_grad():
                    model.logl.fc[0].weight = torch.nn.Parameter(torch.tensor(np.log( (dset.l.T)**2 ))) # dset.l are the true stds, so we convert them to logvar
                # Freeze such weights during the optimization
                for param in model.logl.parameters():
                    param.requires_grad = False
                        
            # Do not train weights of the inference model if we are fixing the sampled latent sources to the true sources
            if args.set_inf:
                for param in model.g.parameters():
                    param.requires_grad = False
                if not config.fix_v:
#                     model.logv.fc[0].requires_grad_(False)  
                    for param in model.logv.parameters():
                        param.requires_grad = False
                

        if not config.ica:
            model = cleanVAE(data_dim=d_data, latent_dim=d_latent, hidden_dim=config.hidden_dim,
                             n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
    
        if config.wandb:
            wandb.watch(model, log='all')

        optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()), history_size=config.history_size, max_iter=config.max_iter, lr=config.lr, line_search_fn='strong_wolfe')


        # load all training data
        Xt, Ut, St, At = dset.x.to(config.device), dset.u.to(config.device), dset.s.to(config.device), torch.from_numpy(dset.A_mix).to(config.device)

        # load all validation data
        val_X, val_U, val_S, val_A = val_dset.x.to(config.device), val_dset.u.to(config.device), val_dset.s.to(config.device), torch.from_numpy(val_dset.A_mix).to(config.device)

        if config.tensorboard:
            # Initialize tensorboard writer with learning seed directory
            writer = SummaryWriter(tensorboard_dir + str(seed) + '/')
            writer.add_graph(model, (Xt, Ut)) # add model to tensorboard
    
        file_path = str(args.run) + '/' + os.path.splitext(args.config)[0] + '_ds_'+str(args.s) + '_ls_' + str(seed) + '_' + str(exp_id) + '.csv'

        if os.path.exists(file_path):
            os.remove(file_path)

        field_names = ['MCC', 'Loss', 'MCC_batch', 'Val_Loss', 'Val_MCC', 'MCS', 'MCS_p', 'Grad', 'Param_diff', 'time']
        append_list_as_row(file_path, field_names)
        
#         st = time.time() # starting time
        for epoch in range(1, config.epochs + 1):
            model.train()

            if config.anneal:
                a = config.a
                d = config.d
                b = config.b
                c = 0
                if epoch > config.epochs / 1.6:
                    b = 1
                    c = 1
                    d = 1
                    a = 2 * config.a
            else:
                a = config.a
                b = config.b
                c = config.c
                d = config.d

            train_loss = 0
            train_perf = 0 # MCC sources
            
            count = 0 # nonlocal variable
            
            st = time.time() # starting time
            # print('data_loader')
            # print(enumerate(data_loader))
            for i, data in enumerate(data_loader):
                print('i', i)
                if not factor:
                    x, u, s_true = data
                else:
                    x, x2, u, s_true = data

                x, u = x.to(config.device), u.to(config.device)
                
                def closure():
                    nonlocal count
                    optimizer.zero_grad()

                    # set seed for sampling z. the loss becomes deterministic.
                    torch.manual_seed(seed)

                    # not tracking the norm of the difference between parameters because it is always 0

                    
                    loss, z = model.elbo(x, u, len(dset))
                    loss.retain_grad()
                    loss.backward(retain_graph=factor)
                    train_loss = loss.item()

                    count = count + 1

                    iteration_time = time.time()
    
                    if config.track_grad_norm:
                        with torch.no_grad():


                            # gradients
                            model_parameters = filter(lambda p: p.requires_grad, model.parameters())

                            grads = [param.grad for param in model_parameters]
                            grads_together = torch.nn.utils.parameters_to_vector(grads)
                            grad_norm = torch.norm(grads_together)
                            grad_norm = grad_norm.item()
                            
                    else:
                        grad_norm = 0
    
                    # validation loss
                    with torch.no_grad():
                        if config.terms:
                            val_loss, val_z, _, _, _ = model.elbo(val_X, val_U, len(val_dset))
                        else:
                            val_loss, val_z = model.elbo(val_X, val_U, len(val_dset), a=a, b=b, c=c, d=d)
                        val_loss = val_loss.item()
                        
                        
                    # MCC score (sources) on the whole training dataset (as opposed to just the training batch)
                    try:
                        # perf_all = mcc(dset.s.numpy(), s.cpu().detach().numpy())
                        perf_all = mcc(dset.s.numpy(), z.cpu().detach().numpy())
                    except:
                        perf_all = 0
                    train_perf = perf_all

                    # MCC (sources) whole validation set
                    try:
                        val_perf = mcc(val_S.cpu().detach().numpy(), val_z.cpu().detach().numpy())
                    except:
                        val_perf = 0

                    # MCS mixing matrix linear assignment
                    try:
                        # mix_perf = mcc(model.f.fc[0].weight.data, At, method='cos')
                        mix_perf = mcc(model.f.fc[0].weight.data.detach().cpu().numpy(), At.detach().cpu().numpy(), method='cos')
                    except:
                        mix_perf = 0

                    # MCS mixing matrix linear permutations
                    if ((config.dd < 11) and (config.dl < 11)):
                        try:
                            # mix_mcs = mcs_p(model.f.fc[0].weight.data, At, method='cos')
                            mix_mcs = mcc(model.f.fc[0].weight.data.detach().cpu().numpy(), At.detach().cpu().numpy(), method='cos')
                        except:
                            mix_mcs = 0
                        if not (config.nps==10):
                            mix_mcs = mix_mcs.item()
                    else:
                        mix_mcs = 0
                    

                    if config.uncentered or config.discrete:
                        if config.dd > 1 and config.dl > 1 and config.ns > 1:
                            # MCC of prior mean
                            est_mean = (model.prior_mean.fc[0].weight.cpu().detach().numpy()).T
                            mcs_mean = mcc(dset.m, est_mean, method='cos')

                    if config.dd > 1 and config.dl > 1 and config.ns > 1:
                        # MCC of prior variance
                        est_var = (model.logl.fc[0].weight.cpu().detach().numpy()).T
                        mcs_var = mcc(dset.l, est_var, method='cos')

                    if config.verbose:
                        print('==> Step {}:\t train loss: {:.6f}\t train perf: {:.6f} \t full perf: {:,.6f} \t mcs lin: {:,.6f} \t mcs p: {:,.6f}'.format(count, train_loss, train_perf, perf_all, mix_perf, mix_mcs))
                        print('==> \t val loss: {:.6f}\t full val perf: {:.6f}'.format(val_loss, val_perf))
                
                    # if config.verbose:
                    #     print('norm diff params', norm_diff)
                    #     print('grad norm', grad_norm)

                    norm_diff = 0

                    row_contents = [perf_all, train_loss, train_perf, val_loss, val_perf, mix_perf, mix_mcs, grad_norm, norm_diff, iteration_time]
                    append_list_as_row(file_path, row_contents)
                    
                    if config.wandb:
                        wandb.log({'Training loss': train_loss, 
                                    'Mixing Matrix MCS Linear Assignment': mix_perf, 
                                    'Mixing Matrix MCS Permutations': mix_mcs, 
                                    'Training Batch MCC (Sources)': train_perf, 
                                    'Training Set MCC (Sources)': perf_all, 
                                    'Validation loss': val_loss, 
                                    'Validation Set MCC (Sources)': val_perf,
                                    'step': count
                                    })
                        if config.dd > 1 and config.dl > 1:
                            wandb.log({'Prior variance MCS': mcs_var, 'step': count})

                        if config.uncentered or config.discrete:
                            if config.dd > 1 and config.dl > 1:
                                wandb.log({'Prior mean MCS': mcs_mean, 'step': count})

                        # The following assumes a square mixing matrix and a linear mixing model
                        # Track mixing model weights
                        for i in range(d_data):
                            for j in range(d_latent): 
                                wandb.log({'f_'+str(i)+str(j): model.f.fc[0].weight[i,j], 'step': count})

                        if config.track_prior:
                            # Track logl weights (do not use, there are too many weights)
                            for i in range(d_latent):
                                for j in range(d_aux):
                                    wandb.log({'logl_'+str(i)+'_'+str(j): model.logl.fc[0].weight[i,j], 'step': count}) # track prior log-var
                                    wandb.log({'prior_mean_'+str(i)+'_'+str(j): model.prior_mean.fc[0].weight[i,j], 'step': count}) # track prior means              

                        if config.terms:
                            # Loss terms
                            wandb.log({'decoder term (mixing model) log p(x|z,u)': logpx, 
                                        'prior term log p(z|u)': logps_cu, 
                                        'encoder term (inference model) log p(z|x,u)': logqs_cux,
                                        'step': count}) 
                            
                        # wandb.log({'norm_diff_params': norm_diff, 'grad_norm': grad_norm, 'step': count})
                            
                    if config.checkpoint:
                        # save checkpoints (weights, loss, performance, meta-data) after every few steps
                        if (count % check_freq == 0):
                            # checkpoint every check_freq count to save space
                            if config.terms:
                                full_checkpoint(ckpt_folder, exp_id, seed, count, model, optimizer, train_loss, train_perf, perf_all, args.s, logpx, logps_cu, logqs_cux, val_loss, val_perf, mix_perf, verbose=config.verbose)
                            else:
                                grads = 0 # to change this
                                checkpoint(ckpt_folder, exp_id, seed, count, model, optimizer, train_loss, train_perf, perf_all, args.s, val_loss, val_perf, mix_perf, grads, iteration_time, verbose=config.verbose)

                        # remove automatic plotting, since it may lead to memory issues
                        # if (count % config.plot_freq == 0):
                        #     fname = os.path.join(args.run, '_'.join([os.path.splitext(args.config)[0], str(args.seed), str(args.n_sims), str(args.s)]))
                        #     ckpt_path = ckpt_folder + str(exp_id) + '/'+ str(exp_id) + '_learn-seed_' + str(seed) + '_data-seed_' + str(args.s) + '_ckpt_'
                        #     if os.path.exists(fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf'):
                        #         os.remove(fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf')

                        #     weights_path = ckpt_path + '*.pth'
                        #     weights_path_last = ckpt_path + str(count) + '.pth'

                        #     if not config.discrete:
                        #         continuous(config=config, dset=test_dset, ckpt=weights_path_last, m_ckpt=weights_path, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf')
                        #     else:
                        #         plot_individual(run=args.run, config=args.config, s=args.s, seed=seed, ckpt_freq=config.check_freq, exp_id=exp_id, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_metrics.pdf', start=1)

                        #         _disc_plots(config=config, dset=test_dset, m_ckpt=weights_path, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf', check_freq=config.check_freq)
                
                    return loss
                
                loss = optimizer.step(closure)
                # print(optimizer.state_dict())
                    
                

#             if config.ica:
#                 if config.discrete or config.uncentered:
#                     f, g, v, s, m, l = model(Xt, Ut) # for discrete_simple_IVAE
#                     #                 _, _, s, _ = model(Xt, Ut) # for DiscreteIVAE
#                 else:
#                     f, g, v, s, l = model(Xt, Ut) # for simple_IVAE
#             else:
#                 _, _, _, s = model(Xt)
        
                    
#             # decide if stop training, tolerance around 1e-4 
#             if norm_diff < config.tol:
#                 break

            if not config.no_scheduler:
                scheduler.step(train_loss)

#             ## early stopping
#             if config.early:
#                 # stop training if no improvement after config.stop epochs
#                 if (stop_count == config.stop):
#                     # checkpoint first
#                     if config.terms:
#                         full_checkpoint(ckpt_folder, exp_id, seed, epoch, model, optimizer, train_loss, train_perf, perf_all, args.s, logpx, logps_cu, logqs_cux, val_loss, val_perf, mix_perf, verbose=config.verbose)
#                     else:
#                         checkpoint(ckpt_folder, exp_id, seed, epoch, model, optimizer, train_loss, train_perf, perf_all, args.s, val_loss, val_perf, mix_perf, iteration_time, verbose=config.verbose)
#                     break
            
        ttime_s = time.time() - st
        print('\ntotal runtime: {} seconds'.format(ttime_s))
        print('\ntotal runtime: {} minutes'.format(ttime_s/60))
        print('\ntotal runtime: {} hours'.format((ttime_s/60)/60))
    
        ## Plots and mixing matrix for every learning seed after the last epoch
        fname = os.path.join(args.run, '_'.join([os.path.splitext(args.config)[0], str(args.seed), str(args.n_sims), str(args.s)]))

        ckpt_path = ckpt_folder + str(exp_id) + '/'+ str(exp_id) + '_learn-seed_' + str(seed) + '_data-seed_' + str(args.s) + '_ckpt_'
        weights_path = ckpt_path + '*.pth'
#         weights_path_last = ckpt_path + str(epoch) + '.pth'
        weights_path_last = ckpt_path + str(count-1) + '.pth' # need to be -1 because the last one is not saved, it breaks before

        # plots of sources only if continuous
        # if not config.discrete:
        #     continuous(config=config, dset=test_dset, ckpt=weights_path_last, m_ckpt=weights_path, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf')
        # else:

        ## do not save pdf files due to memory issues
        # if config.discrete:
        #     plot_individual(run=args.run, config=args.config, s=args.s, seed=seed, ckpt_freq=config.check_freq, exp_id=exp_id, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_metrics.pdf', start=1)
            
        #     disc_plots(config=config, dset=test_dset, m_ckpt=weights_path, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf', check_freq=config.check_freq)
            
        #     inference_plots(m_ckpt=weights_path, config=config, check_freq=config.check_freq, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_inference.pdf')

            # the following does not work for LBFGS!
            # but it would be possible to track the gradient of the weights individually and plot them when we compute the norm
#             if config.dd > 1 and config.dl > 1:
#                 gradient_plots(config=config, dset=test_dset, m_ckpt=weights_path, model=model, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_grads.pdf', check_freq=config.check_freq)

    if config.wandb:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        
    if config.tensorboard:
        writer.close()

    ## remove automatic plotting due to memory issues
    # Analysis plots of all learning seeds together
    # fname2 = os.path.join(args.run, '_'.join([os.path.splitext(args.config)[0], str(args.seed), str(args.n_sims), str(args.s)]))
    # plot_all(run=args.run, config=args.config, s=args.s, seed=args.seed, n_sims=args.n_sims, ckpt_freq=config.check_freq, exp_id=exp_id, pdf_name=fname2+'_'+str(exp_id)+'_analysis.pdf')
    
    