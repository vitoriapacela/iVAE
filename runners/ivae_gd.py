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
    
    # # print true mixing matrix
    # print('-------------------')
    # print('True mixing matrix')
    # print(dset.A_mix)
    # print('-------------------')
    
    # # print true mixing matrix condition number
    # print('Condition number')
    # print(np.linalg.cond(dset.A_mix))
    # print('-------------------')

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
    
#     # Standardize observations
#     if not config.discrete:
#         # Standardize sources before mixing
#         dset.x = (dset.x - torch.mean(dset.x, axis=0))/torch.std(dset.x, axis=0)
#         val_dset.x = (val_dset.x - torch.mean(val_dset.x, axis=0))/torch.std(val_dset.x, axis=0)
#         test_dset.x = (test_dset.x - torch.mean(test_dset.x, axis=0))/torch.std(test_dset.x, axis=0)

    ###

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
#             run_name = run_name.encode('utf-8')
#             print('run_name', run_name)
            wandb.init(project="ivae", config=config, name=run_name, reinit=True, dir=ckpt_folder+'wandb/')
            wandb.config.update(args)

        if config.comet:
            experiment = Experiment(api_key="vnYqg73uIJr8Efrt6gd9TCJU7", project_name="iVAE", workspace="vitoriapacela")
            experiment.log_parameters(config)
            experiment.set_name(str(exp_id)+"_ls_"+str(seed)+"_ds_"+str(args.s))

        if config.ica:
            if config.discrete:
#                 model = DiscreteIVAE(latent_dim=d_latent, data_dim=d_data, aux_dim=d_aux, activation='none', n_layers=config.n_layers, hidden_dim=20, device=config.device)
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
#                             model.f.fc[0].weight[i,j] = torch.nn.Parameter(torch.tensor(dset.A_mix[i,j])) + args.init_noise*(torch.rand(1) - 0.5) # noise from init_noise * U(-0.5, 0.5)
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
                
#             if args.set_g:
#                 # For debugging purposes
#                 # Initialize the inference model (encoder) mean weights to the inverse of the true mixing matrix
#                 A_inv = np.linalg.inv(dset.A_mix)
                
#                 with torch.no_grad():
#                     model.g.fc[0].weight[0,0] = torch.nn.Parameter(torch.tensor(A_inv[0,0]))
#                     model.g.fc[0].weight[0,1] = torch.nn.Parameter(torch.tensor(A_inv[0,1]))
#                     model.g.fc[0].weight[1,0] = torch.nn.Parameter(torch.tensor(A_inv[1,0]))
#                     model.g.fc[0].weight[1,1] = torch.nn.Parameter(torch.tensor(A_inv[1,1]))
                    
#                 # Freeze such weights during the optimization
#                 for param in model.g.parameters():
#                         param.requires_grad = False

        if not config.ica:
            model = cleanVAE(data_dim=d_data, latent_dim=d_latent, hidden_dim=config.hidden_dim,
                             n_layers=config.n_layers, activation=config.activation, slope=.1).to(config.device)
    
        if config.wandb:
            wandb.watch(model, log='all')

#         optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr) # use the filter if you want to fix some parameters
        
#         optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr)

#         if (config.lbfgs) and (not config.gd):
#             optimizer = optim.LBFGS(filter(lambda p: p.requires_grad, model.parameters()), history_size=config.history_size, max_iter=config.max_iter, lr=config.lr, line_search_fn='strong_wolfe')
            
#         elif (config.gd) and (not config.lbfgs):
#              optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr) # use the filter if you want to fix some parameters
                
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr) # use the filter if you want to fix some parameters
                
#         else:
#             raise ValueError("The optimization should be with either GD or LBFGS.")

#         if not config.no_scheduler:
#             # only initialize the scheduler if no_scheduler==False
#             scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True)

#         if factor:
#             D = Discriminator(d_latent).to(config.device)
#             optim_D = optim.Adam(D.parameters(), lr=config.lr,
#                                  betas=(.5, .9))

        # load all training data
        Xt, Ut, St, At = dset.x.to(config.device), dset.u.to(config.device), dset.s.to(config.device), torch.from_numpy(dset.A_mix).to(config.device)

        # load all validation data
        val_X, val_U, val_S, val_A = val_dset.x.to(config.device), val_dset.u.to(config.device), val_dset.s.to(config.device), torch.from_numpy(val_dset.A_mix).to(config.device)

        if config.tensorboard:
            # Initialize tensorboard writer with learning seed directory
            writer = SummaryWriter(tensorboard_dir + str(seed) + '/')
            writer.add_graph(model, (Xt, Ut)) # add model to tensorboard

        # early stopping count
        stop_count = 0
        # early stopping minimum
#         min_loss = 999999
        max_perf = 0 # early stopping max

        st = time.time() # starting time
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
            
            for i, data in enumerate(data_loader):
#                 print('batch', i+1)
#                 print('data[0].shape', data[0].shape)
                if not factor:
                    x, u, s_true = data
                else:
                    x, x2, u, s_true = data

                x, u = x.to(config.device), u.to(config.device)
                
                if config.gd:
                    optimizer.zero_grad()

                    # fix learning seed for sampling operation in the reparemeterization trick
                    torch.manual_seed(seed)

                    if config.terms:
                        # save intermediate loss terms
    #                     loss, z, logpx, logps_cu, logqs_cux = model.elbo(x, u, len(dset))
    #                     if config.norm_z or config.norm_v:
    #                         loss, z, logpx, logps_cu, logqs_cux, pen_var, pen_mu = model.elbo(x, u, len(dset))
    #                     else: 
                        loss, z, logpx, logps_cu, logqs_cux = model.elbo(x, u, len(dset))

                        # set latent sources to true sources
                        if args.set_inf:
                            z = s_true

                    else:
    #                     loss, z = model.elbo(x, u) # for DiscreteIVAE
                        loss, z = model.elbo(x, u, len(dset), a=a, b=b, c=c, d=d)

    #                 if factor:
    #                     D_z = D(z)
    #                     vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean()
    #                     loss += config.gamma * vae_tc_loss

                    # compute gradient
                    loss.retain_grad()
                    loss.backward(retain_graph=factor)

                    # Normalize mixing model
                    # Only defined for when the mixing model is a linear transformation!
                    # That is, no hidden layers, n_layers=1 !!!
                    if config.norm_f:
                        with torch.no_grad():
                            h = model.f.fc[0].weight.clone().detach()
                            model.f.fc[0].weight[:] = torch.nn.Parameter(h.div_(torch.norm(h, dim=0, keepdim=True))) # columns  
                        
                    if config.norm_logl:
                        # normalize prior log-variances
                        with torch.no_grad():
    #                         h = model.logl.fc[0].weight.clone().detach()
    #                         model.logl.fc[0].weight[:] = torch.nn.Parameter(h.div_(torch.norm(h, dim=0, keepdim=True))) # column-wise. Do not use. There is scale indeterminacy in the mixing matrix which is connected to a variance of a single source over all segments, not all sources over one segment.

                            ## Use this usually!
                            h = model.logl.fc[0].weight.clone().detach()
                            model.logl.fc[0].weight[:] = torch.nn.Parameter(h.div_(torch.norm(h, dim=1, keepdim=True))) # row-wise

                            # normalize the variance and project to log-variance
    #                         h = model.logl.fc[0].weight.clone().detach()
    #                         var = h.exp()
    #                         norm_var = var.div_( torch.norm(var, dim=1, keepdim=True) )
    #                         norm_logl = norm_var.log()
    #                         model.logl.fc[0].weight[:] = torch.nn.Parameter(norm_logl)

    #                         h_ = model.logl.fc[0].weight.clone().detach()
    #                         model.logl.fc[0].weight[:] = torch.nn.Parameter(h_.div_(torch.norm(h_, keepdim=True))) # Frobenius (all matrix)
    
                    if config.norm_prior_mean:
                            # normalize prior means
                        with torch.no_grad():
    #                         h_ = model.prior_mean.fc[0].weight.clone().detach()
    #                         model.prior_mean.fc[0].weight[:] = torch.nn.Parameter(h_.div_(torch.norm(h_, dim=0, keepdim=True))) # column-wise. Do NOT use. There is scale indeterminacy in the mixing matrix which is connected to a variance of a single source over all segments, not all sources over one segment.

                            h_ = model.prior_mean.fc[0].weight.clone().detach()
                            model.prior_mean.fc[0].weight[:] = torch.nn.Parameter(h_.div_(torch.norm(h_, dim=1, keepdim=True))) # row-wise

    #                         h_ = model.prior_mean.fc[0].weight.clone().detach()
    #                         model.prior_mean.fc[0].weight[:] = torch.nn.Parameter(h_.div_(torch.norm(h_, keepdim=True))) # Frobenius (all matrix)
                

                    # batch loss
                    train_loss += loss.item()
                
                    # MCC sources (training batch)
                    try:
                        perf = mcc(s_true.numpy(), z.cpu().detach().numpy())
                        # why does this work? should be 
                        # perf = mcc(s_true.cpu().numpy(), z.cpu().detach().numpy())
                    except:
                        perf = 0
                    train_perf += perf

#                 elif config.lbfgs:
#                     def closure():
#                         optimizer.zero_grad()

#                         # set seed for sampling z. the loss becomes deterministic.
#                         torch.manual_seed(seed)

#                         loss, z = model.elbo(x, u, len(dset))
#                         loss.retain_grad()
#                         loss.backward(retain_graph=factor)
#                         return loss

                with torch.no_grad():
                    # track parameters here
                    # parameters before the update
                    all_param = [m.data for m in model.parameters()]
#                     all_param = filter(lambda p: p.requires_grad, model.parameters()) # as it is, this generates an error
                    vec = torch.nn.utils.parameters_to_vector(all_param)
                
#                 if config.lbfgs:
#                     loss = optimizer.step(closure)
# #                     print(optimizer.state_dict())
#                     train_loss += loss.item()
                    
#                 elif config.gd:
#                     optimizer.step()
                
                optimizer.step()
                
                with torch.no_grad():
                    # parameters after the update
                    new_param = [m.data for m in model.parameters()]
                    new_vec = torch.nn.utils.parameters_to_vector(all_param)
    #                 print('new_vec', new_vec)

                    norm_diff = torch.norm(new_vec - vec)
                    norm_diff = norm_diff.item()

                    # gradients
                    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #                 grads = [param.grad for param in model.parameters()]
                    grads = [param.grad for param in model_parameters]
                    grads_together = torch.nn.utils.parameters_to_vector(grads)
                    grad_norm = torch.norm(grads_together)
                    grad_norm = grad_norm.item()
            
#                     if config.verbose==True:
#                         print('norm diff params', norm_diff)
#                         print('grad norm', grad_norm)
        
                if config.wandb:
                    wandb.log({'norm_diff_params': norm_diff, 'grad_norm': grad_norm, 'epoch': epoch})
#                     wandb.log({'norm_diff_params': norm_diff, 'grad_norm': grad_norm}, step=epoch)
                
#                 if factor:
#                     ones = torch.ones(config.batch_size, dtype=torch.long, device=config.device)
#                     zeros = torch.zeros(config.batch_size, dtype=torch.long, device=config.device)
#                     x_true2 = x2.to(config.device)
#                     _, _, _, z_prime = model(x_true2)
#                     z_pperm = permute_dims(z_prime).detach()
#                     D_z_pperm = D(z_pperm)
#                     D_tc_loss = 0.5 * (F.cross_entropy(D_z, zeros) + F.cross_entropy(D_z_pperm, ones))

#                     optim_D.zero_grad()
#                     D_tc_loss.backward()
#                     optim_D.step()

            train_perf /= len(data_loader)
            train_loss /= len(data_loader)

            # save results on the whole validation set
            with torch.no_grad():
                if config.terms:
#                     val_loss, val_z, _, _, _ = model.elbo(val_X, val_U, len(val_dset), a=a, b=b, c=c, d=d)
#                     if config.norm_z or config.norm_v:
#                         val_loss, val_z, _, _, _, _, _ = model.elbo(val_X, val_U, len(val_dset))
#                     else:
                    val_loss, val_z, _, _, _ = model.elbo(val_X, val_U, len(val_dset))
                else:
#                     val_loss, val_z = model.elbo(val_X, val_U) # for DiscreteIVAE
                    val_loss, val_z = model.elbo(val_X, val_U, len(val_dset), a=a, b=b, c=c, d=d)
                val_loss = val_loss.item()

#             # for early stopping monitoring the validation loss
#             if config.early:
#                 if (val_loss.item() > min_loss):
#                     stop_count += 1

#                 elif (val_loss.item() < min_loss):
#                     min_loss = val_loss.item()
#                     stop_count = 0


            if config.ica:
                if config.discrete or config.uncentered:
#                     print('Xt', Xt.size())
#                     print('--')
#                     print('Ut', Ut.size())
                    f, g, v, s, m, l = model(Xt, Ut) # for discrete_simple_IVAE
                    #                 _, _, s, _ = model(Xt, Ut) # for DiscreteIVAE
                else:
                    f, g, v, s, l = model(Xt, Ut) # for simple_IVAE
            else:
                _, _, _, s = model(Xt)

            # MCC score (sources) on the whole training dataset (as opposed to just the training batch)
            try:
                perf_all = mcc(dset.s.numpy(), s.cpu().detach().numpy())
            except:
                perf_all = 0

            # MCC (sources) whole validation set
            try:
                val_perf = mcc(val_S.cpu().detach().numpy(), val_z.cpu().detach().numpy())
            except:
                val_perf = 0

            # MCS mixing matrix linear assignment
            try:
#                 mix_perf = mcc(model.f.fc[0].weight.data, At, method='cos')
                mix_perf = mcc(model.f.fc[0].weight.data.detach().cpu().numpy(), At.detach().cpu().numpy(), method='cos')
            except:
                mix_perf = 0
            
            # MCS mixing matrix linear permutations
            # try:
            #     # mix_mcs = mcs_p(model.f.fc[0].weight.data, At, method='cos')
            #     mix_mcs = mcc(model.f.fc[0].weight.data.detach().cpu().numpy(), At.detach().cpu().numpy(), method='cos')
            # except:
            #     mix_mcs = 0
            # mix_mcs = mix_mcs.item()
            
            iteration_time = time.time() - st

            file_path = str(args.run) + '/' + os.path.splitext(args.config)[0] + '_ds_'+str(args.s) + '_ls_' + str(seed) + '_' + str(exp_id) + '.csv'
            if epoch==1:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    
                # field_names = ['MCC', 'Loss', 'MCC_batch', 'Val_Loss', 'Val_MCC', 'MCS', 'MCS_p', 'Grad', 'Param_diff', 'time']
                field_names = ['MCC', 'Loss', 'MCC_batch', 'Val_Loss', 'Val_MCC', 'MCS', 'Grad', 'Param_diff', 'time']
                append_list_as_row(file_path, field_names)
            
            # row_contents = [perf_all, train_loss, train_perf, val_loss, val_perf, mix_perf, mix_mcs, grad_norm, norm_diff, iteration_time]
            row_contents = [perf_all, train_loss, train_perf, val_loss, val_perf, mix_perf, grad_norm, norm_diff, iteration_time]
            append_list_as_row(file_path, row_contents)

#             file_path_e = str(args.run) + '/' + os.path.splitext(args.config)[0] + '_ds_'+str(args.s) + '_ls_' + str(seed) + '_' + str(exp_id) + '_epoch_' + str(epoch) + '.csv'
#             append_list_as_row(file_path_e, row_contents)
        
            # for early stopping monitoring the mixing matrix mcc
            if config.early:
                if (mix_perf < max_perf):
                    stop_count += 1

                elif (mix_perf > max_perf):
                    max_perf = mix_perf
                    stop_count = 0

            # Check mean and variance of the estimated sources for each component
            z_means = torch.mean(s, axis=0)
            z_vars = torch.var(s, axis=0)
            
            x_means = torch.mean(f, axis=0)
            x_vars = torch.var(f, axis=0)
            
            
            # Old implementation: this was used to compute the MCC of means and variances across all the datapoints
#             # True prior variance per data point
#             L_var = np.zeros((len(dset.x), config.dl))
#             for seg in range(len(dset.l)):
#                 segID = range(config.nps * seg, config.nps * (seg + 1))
#                 L_var[segID] = dset.l[seg]
                
            if config.uncentered or config.discrete:
#                 # True prior mean per data point
#                 m_seg = np.zeros((len(dset.x), config.dl))
#                 for seg in range(len(dset.m)):
#                     segID = range(config.nps * seg, config.nps * (seg + 1))
#                     m_seg[segID] = dset.m[seg]
                
                if config.dd > 1 and config.dl > 1 and config.ns > 1:
                    # MCC of prior mean
                    est_mean = (model.prior_mean.fc[0].weight.cpu().detach().numpy()).T
                    # print("est mean", est_mean)
                    mcs_mean = mcc(dset.m, est_mean, method='cos')
#                     print('mcs_mean', mcs_mean)
            
            if config.dd > 1 and config.dl > 1 and config.ns > 1:
                # MCC of prior variance
                est_var = (model.logl.fc[0].weight.cpu().detach().numpy()).T
                # print("est var", est_var)
                mcs_var = mcc(dset.l, est_var, method='cos')
#                 print('mcs_var', mcs_var)
            
            if config.verbose:
                # print('==> Epoch {}/{}:\t train loss: {:.6f}\t train perf: {:.6f} \t full perf: {:,.6f} \t mix mcc: {:,.6f} \t mix mcs: {:,.6f}'.format(epoch, config.epochs, train_loss, train_perf, perf_all, mix_perf, mix_mcs))
                print('==> Epoch {}/{}:\t train loss: {:.6f}\t train perf: {:.6f} \t full perf: {:,.6f} \t mix mcs: {:,.6f}'.format(epoch, config.epochs, train_loss, train_perf, perf_all, mix_perf))
                print('==> \t val loss: {:.6f}\t full val perf: {:.6f}'.format(val_loss, val_perf))

            # Check norm of each component of the mixing model
            norms = torch.norm(model.f.fc[0].weight, dim=0)
        
            if config.wandb:
                wandb.log({'Training loss': train_loss, 
                           'Mixing Matrix MCS Linear Assignment': mix_perf, 
                        #    'Mixing Matrix MCS Permutations': mix_mcs, 
                           'Training Batch MCC (Sources)': train_perf, 
                           'Training Set MCC (Sources)': perf_all, 
                           'Validation loss': val_loss, 
                           'Validation Set MCC (Sources)': val_perf,
                           'epoch': epoch
                          })
                if config.dd > 1 and config.dl > 1:
                    wandb.log({'Prior variance MCS': mcs_var, 'epoch': epoch})
                    
                if config.uncentered or config.discrete:
                    if config.dd > 1 and config.dl > 1:
                        wandb.log({'Prior mean MCS': mcs_mean, 'epoch': epoch})
                
                # The following assumes a square mixing matrix and a linear mixing model
                # Track mixing model weights
                for i in range(d_data):
                    for j in range(d_latent): 
                        wandb.log({'f_'+str(i)+str(j): model.f.fc[0].weight[i,j], 'epoch': epoch})
                
                if config.track_prior:
                    # Track logl weights (do not use, there are too many weights)
                    for i in range(d_latent):
                        for j in range(d_aux):
                            wandb.log({'logl_'+str(i)+'_'+str(j): model.logl.fc[0].weight[i,j], 'epoch': epoch}) # track prior log-var
                            wandb.log({'prior_mean_'+str(i)+'_'+str(j): model.prior_mean.fc[0].weight[i,j], 'epoch': epoch}) # track prior means              
                
                for i in range(len(z_means)):
                    wandb.log({'mu_z_'+str(i): z_means[i],
                               'var_z_'+str(i): z_vars[i], 'norm_f_'+str(i): norms[i],
                               'epoch': epoch
                              })

                if config.terms:
                    # Loss terms
                    wandb.log({'decoder term (mixing model) log p(x|z,u)': logpx, 
                               'prior term log p(z|u)': logps_cu, 
                               'encoder term (inference model) log p(z|x,u)': logqs_cux,
                               'epoch': epoch})
                    
            # decide if stop training, tolerance around 1e-4 
            if norm_diff < config.tol:
                break

            if config.tensorboard:
                # Track training Tensorboard
                writer.add_scalar('Training loss', train_loss, epoch)
                writer.add_scalar('Mixing Matrix MCS', mix_perf, epoch)
                writer.add_scalar('Training Batch MCC (Sources)', train_perf, epoch)
                writer.add_scalar('Training Set MCC (Sources)', perf_all, epoch)
                writer.add_scalar('Validation loss', val_loss, epoch)
                writer.add_scalar('Validation Set MCC (Sources)', val_perf, epoch)

                # print loss gradient, but it should be always 1
    #           print('loss grad: ', loss.grad)

                # gradient for each parameter
                for name, param in model.named_parameters():
                    if param.grad is not None:
    #                     print(name, param.grad.mean())
                        writer.add_scalar('Mean ' + name, param.grad.mean(), epoch)
                        writer.add_scalar('Var ' + name, param.grad.var(), epoch)
    #                 else:
    #                     print(name, param.grad)

                # clip weights in case of exploding gradients
    #           clip_grad_norm(model.parameters(), 2)

                # Save mixing matrix weights to Tensorboard
                writer.add_scalar('f_00', model.f.fc[0].weight[0,0], epoch)
                writer.add_scalar('f_01', model.f.fc[0].weight[0,1], epoch)
                writer.add_scalar('f_10', model.f.fc[0].weight[1,0], epoch)
                writer.add_scalar('f_11', model.f.fc[0].weight[1,1], epoch)

                if config.terms:
                    # Loss terms to tensorboard
                    writer.add_scalar('decoder term (mixing model), log p(x|z,u)', logpx, epoch)
                    writer.add_scalar('prior term, log p(z|u)', logps_cu, epoch)
                    writer.add_scalar('encoder term (inference model), log p(z|x,u)', logqs_cux, epoch)

            if config.comet:
                # Track training with Comet ML
                experiment.log_metric('Training loss', train_loss, epoch)
                experiment.log_metric('Mixing Matrix MCS (linear assignment)', mix_perf, epoch)
                experiment.log_metric('Training Batch MCC (Sources)', train_perf, epoch)
                experiment.log_metric('Training Set MCC (Sources)', perf_all, epoch)
                experiment.log_metric('Validation loss', val_loss, epoch)
                experiment.log_metric('Validation Set MCC (Sources)', val_perf, epoch)
                # experiment.log_metric('Mixing matrix MCS (permutations)', mix_mcs, epoch)

                # gradient for each parameter
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        experiment.log_metric('Mean ' + name, param.grad.mean(), epoch)
                        experiment.log_metric('Var ' + name, param.grad.var(), epoch)

                # clip weights in case of exploding gradients
    #           clip_grad_norm(model.parameters(), 2)

                # Mixing matrix weights
                experiment.log_metric('f_00', model.f.fc[0].weight[0,0], epoch)
                experiment.log_metric('f_01', model.f.fc[0].weight[0,1], epoch)
                experiment.log_metric('f_10', model.f.fc[0].weight[1,0], epoch)
                experiment.log_metric('f_11', model.f.fc[0].weight[1,1], epoch)

                if config.terms:
                    # Loss terms
                    experiment.log_metric('decoder term (mixing model), log p(x|z,u)', logpx, epoch)
                    experiment.log_metric('prior term, log p(z|u)', logps_cu, epoch)
                    experiment.log_metric('encoder term (inference model), log p(z|x,u)', logqs_cux, epoch)

            if config.checkpoint:
                # save checkpoints (weights, loss, performance, meta-data) after every epoch
                if (epoch % check_freq == 0):
                    # checkpoint every check_freq epochs to save space
                    if config.terms:
                        full_checkpoint(ckpt_folder, exp_id, seed, epoch, model, optimizer, train_loss, train_perf, perf_all, args.s, logpx, logps_cu, logqs_cux, val_loss, val_perf, mix_perf, verbose=config.verbose)
                    else:
                        checkpoint(ckpt_folder, exp_id, seed, epoch, model, optimizer, train_loss, train_perf, perf_all, args.s, val_loss, val_perf, mix_perf, grads, iteration_time, verbose=config.verbose)
                        
                # if (epoch % config.plot_freq == 0):
                #     fname = os.path.join(args.run, '_'.join([os.path.splitext(args.config)[0], str(args.seed), str(args.n_sims), str(args.s)]))
                #     ckpt_path = ckpt_folder + str(exp_id) + '/'+ str(exp_id) + '_learn-seed_' + str(seed) + '_data-seed_' + str(args.s) + '_ckpt_'
                #     if os.path.exists(fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf'):
                #         os.remove(fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf')
                        
                #     weights_path = ckpt_path + '*.pth'
                #     weights_path_last = ckpt_path + str(epoch) + '.pth'
                    
                #     if not config.discrete:
                #         continuous(config=config, dset=test_dset, ckpt=weights_path_last, m_ckpt=weights_path, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf')
                #     else:
                #         plot_individual(run=args.run, config=args.config, s=args.s, seed=seed, ckpt_freq=config.check_freq, exp_id=exp_id, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_metrics.pdf', start=1)
                        
                #         _disc_plots(config=config, dset=test_dset, m_ckpt=weights_path, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf', check_freq=config.check_freq)

            if not config.no_scheduler:
                scheduler.step(train_loss)

            ## early stopping
            if config.early:
                # stop training if no improvement after config.stop epochs
                if (stop_count == config.stop):
                    # checkpoint first
                    if config.terms:
                        full_checkpoint(ckpt_folder, exp_id, seed, epoch, model, optimizer, train_loss, train_perf, perf_all, args.s, logpx, logps_cu, logqs_cux, val_loss, val_perf, mix_perf, verbose=config.verbose)
                    else:
                        checkpoint(ckpt_folder, exp_id, seed, epoch, model, optimizer, train_loss, train_perf, perf_all, args.s, val_loss, val_perf, mix_perf, iteration_time, verbose=config.verbose)
                    break
            
        ttime_s = time.time() - st
        print('\ntotal runtime: {} seconds'.format(ttime_s))
        print('\ntotal runtime: {} minutes'.format(ttime_s/60))
        print('\ntotal runtime: {} hours'.format((ttime_s/60)/60))
    
        ## Plots and mixing matrix for every learning seed after the last epoch
        fname = os.path.join(args.run, '_'.join([os.path.splitext(args.config)[0], str(args.seed), str(args.n_sims), str(args.s)]))

        ckpt_path = ckpt_folder + str(exp_id) + '/'+ str(exp_id) + '_learn-seed_' + str(seed) + '_data-seed_' + str(args.s) + '_ckpt_'
        weights_path = ckpt_path + '*.pth'
#         weights_path_last = ckpt_path + str(epoch) + '.pth'
        weights_path_last = ckpt_path + str(epoch-1) + '.pth' # need to be -1 because the last one is not saved, it breaks before

        # plots of sources only if continuous
        if config.discrete:
            plot_individual(run=args.run, config=args.config, s=args.s, seed=seed, ckpt_freq=config.check_freq, exp_id=exp_id, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_metrics.pdf', start=1)
            
            disc_plots(config=config, dset=test_dset, m_ckpt=weights_path, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_plots.pdf', check_freq=config.check_freq)
            
            inference_plots(m_ckpt=weights_path, config=config, check_freq=config.check_freq, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_inference.pdf')
    
            if config.dd > 1 and config.dl > 1:
                gradient_plots(config=config, dset=test_dset, m_ckpt=weights_path, model=model, pdf_name=fname+'_ls_'+str(seed)+'_'+str(exp_id)+'_grads.pdf', check_freq=config.check_freq)

    if config.wandb:
        torch.save(model.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
        
    if config.tensorboard:
        writer.close()

    ## remove automatic plotting due to memory issues
    # # Analysis plots of all learning seeds together
    # fname2 = os.path.join(args.run, '_'.join([os.path.splitext(args.config)[0], str(args.seed), str(args.n_sims), str(args.s)]))
    # plot_all(run=args.run, config=args.config, s=args.s, seed=args.seed, n_sims=args.n_sims, ckpt_freq=config.check_freq, exp_id=exp_id, pdf_name=fname2+'_'+str(exp_id)+'_analysis.pdf')
    
    
    