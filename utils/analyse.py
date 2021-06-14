import matplotlib.pyplot as plt
import numpy as np
import torch
import glob
import os
import sys
from scipy.stats import sem
import pandas as pd

from .plots import tol_mean_sem

from matplotlib.backends.backend_pdf import PdfPages

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.set_default_dtype(torch.double)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    
def plot_all(run, config, s, seed, n_sims, ckpt_freq, exp_id, pdf_name='plots.pdf', start=10):
    '''
    Read iVAE runner data, which contains the following items:
    Training batch MCC Sources, Training batch loss, Training set MCC sources, Validation loss, MCC Sources validation set, Mixing matrix MCS linear assignment, Mixing matrix MCS permutations, time. Each item per epoch for each learning seed.
    All this data refers to only one data seed.
    Create multiple plots comparing how each learning seed performs for each item, and also plotting the mean of each item across all learning seeds.
    Save all the plots to a pdf.
    
    run: (str) path where the run is logged ('run/' by default)
    s: (int) data seed
    seed: (int) initial learning seed
    n_sims: (int) total number of simulations (learning seeds)
    ckpt_freq: (int) checkpoint frequency
    exp_id: (int) experiment ID
    pdf_name: (str) path to save the pdf containing the plots.
    start: (int) start epoch for plots
    '''
    # checkpoint frequency
    check_freq = 1 # otherwise, not implemented!!!
#     check_freq = ckpt_freq
    
    # go over each learning seed, each one has a different number of epochs
    all_seeds = [] # list of dataframes for each learning seed
    for ls in range(seed, seed + n_sims):
        file_path = str(run) + '/' + os.path.splitext(config)[0] + '_ds_' + str(s) + '_ls_' + str(ls) + '_' + str(exp_id)  + '.csv'
        df = pd.read_csv(file_path)
        all_seeds.append(df)
        
    # take the maximum for max_epoch    
    epoch_seed = []
    for i in range(len(all_seeds)):
        n_epochs = len(all_seeds[i]['MCC'])
        epoch_seed.append(n_epochs)
    max_epoch = max(epoch_seed)
    
    all_full_mcc = [] # MCC sources full training set
    all_losses = [] # training batch loss
    all_mcc = [] # training batch MCC sources
    all_val_losses = [] # validation loss
    all_val_mcc = [] # validation MCC sources
    all_mix_mcc = [] # MCS mixing matrix linear assignment (rename variable?)
    all_mix_mcs = [] # MCS mixing matrix permutations
    all_grads = [] # norm of gradients
    all_norm_diff = [] # norm of the difference between current and previous parameters
    all_time = [] # running time
    
    for i in range(len(all_seeds)):
        all_full_mcc.append(all_seeds[i]['MCC'].values)
        all_losses.append(all_seeds[i]['Loss'].values)
        all_mcc.append(all_seeds[i]['MCC_batch'].values)
        all_val_losses.append(all_seeds[i]['Val_Loss'].values)
        all_val_mcc.append(all_seeds[i]['Val_MCC'].values)
        all_mix_mcc.append(all_seeds[i]['MCS'].values)
        all_mix_mcs.append(all_seeds[i]['MCS_p'].values)
        all_grads.append(all_seeds[i]['Grad'].values)
        all_norm_diff.append(all_seeds[i]['Param_diff'].values)
        all_time.append(all_seeds[i]['time'].values)

    loss_mean, loss_error = tol_mean_sem(all_losses, max_epoch) # Average training loss 
    val_loss_mean, val_loss_error = tol_mean_sem(all_val_losses, max_epoch) # Average validation loss
    mcc_mean, mcc_error = tol_mean_sem(all_mcc, max_epoch) # Average training MCC (batch) sources
    all_mcc_mean, all_mcc_error = tol_mean_sem(all_full_mcc, max_epoch) # Average training MCC (all training set)
    val_mcc_mean, val_mcc_error = tol_mean_sem(all_val_mcc, max_epoch) # Average validation MCC (sources)
    mix_mcc_mean, mix_mcc_error = tol_mean_sem(all_mix_mcc, max_epoch) # Average MCS (mixing matrix) - linear assignment
    mix_mcs_mean, mix_mcs_error = tol_mean_sem(all_mix_mcs, max_epoch) # Average MCS (mixing matrix) - permutations
    
    
    pp = PdfPages(pdf_name)

    ## Individual learning seeds

    plot1 = plt.figure()
    
    # plt.text(0.5, 1.5, "Individual seeds", size=20)
    for losses_i in all_losses:
        max_i = len(losses_i)
        zoom_i = np.arange(start, max_i)
        plt.plot(zoom_i, losses_i[start:])
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Training batch loss')
    pp.savefig(plot1)

    plot2 = plt.figure()
    for val_losses_i in all_val_losses:
        max_i = len(val_losses_i)
        zoom_i = np.arange(start, max_i)
        plt.plot(zoom_i, val_losses_i[start:])
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Validation loss')
    pp.savefig(plot2)

#     plotm = plt.figure(figsize=(10,10))
#     for val_losses_i in all_val_losses:
#         plt.plot(x_range, val_losses_i)
#     plt.xlabel('Epoch')
#     plt.ylabel('ELBO')
#     plt.title('Validation loss')
#     plt.ylim(1.39, 1.44)
#     pp.savefig(plotm)

    plot3 = plt.figure()
#     for all_losses_i in all_losses:
    for i in range(len(all_losses)):
        all_losses_i = all_losses[i]
        max_i = len(all_losses_i)
        zoom_i = np.arange(start, max_i)
        plt.plot(zoom_i, all_losses_i[start:], label=i)
        plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Training batch loss for different simulation seeds')
    pp.savefig(plot3)
    plt.close()

    plot4 = plt.figure()
    for mcc_i in all_mcc:
        max_i = len(mcc_i)
        x_range_i = np.arange(1, max_i+1)
        x_range_i = x_range_i[x_range_i % check_freq == 0]
        plt.plot(x_range_i, mcc_i)
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - MCC - Training batch')
    pp.savefig(plot4)
    plt.close()

    plot5 = plt.figure()
    for all_mcc_i in all_full_mcc:
        max_i = len(all_mcc_i)
        x_range_i = np.arange(1, max_i+1)
        x_range_i = x_range_i[x_range_i % check_freq == 0]
        plt.plot(x_range_i, all_mcc_i)
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - MCC training set')
    pp.savefig(plot5)
    plt.close()

    plot6 = plt.figure()
    for val_mcc_i in all_val_mcc:
        max_i = len(val_mcc_i)
        x_range_i = np.arange(1, max_i+1)
        x_range_i = x_range_i[x_range_i % check_freq == 0]
        plt.plot(x_range_i, val_mcc_i)
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - MCC validation set')
    pp.savefig(plot6)
    plt.close()

    plot7 = plt.figure()
    for mix_mcc_i in all_mix_mcc:
        max_i = len(mix_mcc_i)
        x_range_i = np.arange(1, max_i+1)
        x_range_i = x_range_i[x_range_i % check_freq == 0]
        plt.plot(x_range_i, mix_mcc_i)
    plt.xlabel('Epoch')
    plt.ylabel('MCS')
    plt.title('Mixing matrix MCS (linear assignment)')
    pp.savefig(plot7)
    plt.close()
    
    plot7a = plt.figure()
    for i in range(len(all_mix_mcc)):
        plt.plot(all_time[i], all_mix_mcc[i])
    plt.xlabel('Time (s)')
    plt.ylabel('MCS')
    plt.title('Mixing matrix MCS (linear assignment)')
    pp.savefig(plot7a)
    plt.close()
    
    plot7b = plt.figure()
    for mix_mcs_i in all_mix_mcs:
        max_i = len(mix_mcs_i)
        x_range_i = np.arange(1, max_i+1)
        x_range_i = x_range_i[x_range_i % check_freq == 0]
        plt.plot(x_range_i, mix_mcs_i)
    plt.xlabel('Epoch')
    plt.ylabel('MCS')
    plt.title('Mixing matrix MCS (permutations)')
    pp.savefig(plot7b)
    plt.close()
    
    plot7c = plt.figure()
    for grad_i in all_grads:
        max_i = len(grad_i)
        x_range_i = np.arange(1, max_i+1)
        x_range_i = x_range_i[x_range_i % check_freq == 0]
        plt.plot(x_range_i, grad_i)
    plt.xlabel('Epoch')
    plt.ylabel(r'$\vert \vert \nabla \vert \vert$')
    plt.title('Norm of the gradients')
    pp.savefig(plot7c)
    plt.close()
    
    plot7d = plt.figure()
    for diff_i in all_norm_diff:
        max_i = len(diff_i)
        x_range_i = np.arange(1, max_i+1)
        x_range_i = x_range_i[x_range_i % check_freq == 0]
        plt.plot(x_range_i, diff_i)
    plt.xlabel('Epoch')
    plt.ylabel(r'$\vert \vert \theta_i - \theta_{i-1} \vert \vert$')
    plt.title('Norm of the difference between current and previous parameters')
    pp.savefig(plot7d)
    plt.close()

    ## Mean across all learning seeds
    x_range = np.arange(1, max_epoch+1)
    x_range = x_range[x_range % check_freq == 0]

    plot8 = plt.figure(figsize=(9, 4.8))
    # plt.text(0.5, 1.5, "Mean across all learning seeds", size=20)
    plt.plot(x_range, loss_mean[0:max_epoch], color='pink')
    plt.fill_between(x_range, loss_mean[0:max_epoch] - loss_error[0:max_epoch], loss_mean[0:max_epoch] + loss_error[0:max_epoch], alpha=0.3, color='pink')
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Average training batch loss')
    plt.yscale('log')
    pp.savefig(plot8)
    plt.close()

    # Zoom in
    plot9 = plt.figure()
    plt.plot(np.arange(start, max_epoch), loss_mean[start:max_epoch], color='pink')
    plt.fill_between(np.arange(start, max_epoch), loss_mean[start:max_epoch] - loss_error[start:max_epoch], loss_mean[start:max_epoch] + loss_error[start:max_epoch], alpha=0.3, color='pink')
    plt.xlim(start, max_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Average training batch loss')
    pp.savefig(plot9)
    plt.close()

    plot10 = plt.figure()
    plt.plot(x_range, val_loss_mean[0:max_epoch], color='purple')
    plt.fill_between(x_range, val_loss_mean[0:max_epoch] - val_loss_error[0:max_epoch], val_loss_mean[0:max_epoch] + val_loss_error[0:max_epoch], alpha=0.3, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Average validation loss')
    pp.savefig(plot10)
    plt.close()

    # Zoom in
    zoom_range = np.arange(start, max_epoch)
    zoom_range = zoom_range[zoom_range % check_freq == 0]
    plot11 = plt.figure()
    plt.plot(zoom_range, val_loss_mean[start:max_epoch], color='purple')
    plt.fill_between(zoom_range, val_loss_mean[start:max_epoch] - val_loss_error[start:max_epoch], val_loss_mean[start:max_epoch] + val_loss_error[start:max_epoch], alpha=0.3, color='purple')
    plt.xlim(start, max_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Average validation loss')
    pp.savefig(plot11)
    plt.close()

    plot12 = plt.figure()
    plt.plot(x_range, loss_mean[0:max_epoch], label='Training', color='pink')
    plt.fill_between(x_range, loss_mean[0:max_epoch] - loss_error[0:max_epoch], loss_mean[0:max_epoch] + loss_error[0:max_epoch], alpha=0.3, color='pink')
    plt.plot(x_range, val_loss_mean[0:max_epoch], label='Validation', color='purple')
    plt.fill_between(x_range, val_loss_mean[0:max_epoch] - val_loss_error[0:max_epoch], val_loss_mean[0:max_epoch] + val_loss_error[0:max_epoch], alpha=0.3, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Average Loss')
    plt.legend()
    pp.savefig(plot12)
    plt.close()

    plot13 = plt.figure(figsize=(7, 4.8))
    # Zoom
#     zoom_range = np.arange(start, max_epoch)
    plt.plot(zoom_range, loss_mean[start:max_epoch], label='Training', color='pink')
    plt.fill_between(zoom_range, loss_mean[start:max_epoch] - loss_error[start:max_epoch], loss_mean[start:max_epoch] + loss_error[start:max_epoch], alpha=0.3, color='pink')
    plt.plot(zoom_range, val_loss_mean[start:max_epoch], label='Validation', color='purple')
    plt.fill_between(zoom_range, val_loss_mean[start:max_epoch] - val_loss_error[start:max_epoch], val_loss_mean[start:max_epoch] + val_loss_error[start:max_epoch], alpha=0.3, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Average Loss')
    plt.legend()
    plt.yscale('log')
    pp.savefig(plot13)
    plt.close()

    plot14 = plt.figure()
    plt.plot(x_range, mcc_mean[0:max_epoch], color='pink')
    plt.fill_between(x_range, mcc_mean[0:max_epoch] - mcc_error[0:max_epoch], mcc_mean[0:max_epoch] + mcc_error[0:max_epoch], alpha=0.3, color='pink')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - Average training batch performance')
    pp.savefig(plot14)
    plt.close()

    plot15 = plt.figure()
    # Zoom in
    plt.plot(zoom_range, mcc_mean[start:max_epoch], color='pink')
    plt.fill_between(zoom_range, mcc_mean[start:max_epoch] - mcc_error[start:max_epoch], mcc_mean[start:max_epoch] + mcc_error[start:max_epoch], alpha=0.3, color='pink')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - Average training batch performance')
    pp.savefig(plot15)
    plt.close()

    plot16 = plt.figure()
    plt.plot(x_range, all_mcc_mean[0:max_epoch], color='pink')
    plt.fill_between(x_range, all_mcc_mean[0:max_epoch] - all_mcc_error[0:max_epoch], all_mcc_mean[0:max_epoch] + all_mcc_error[0:max_epoch], alpha=0.3, color='pink')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - Average performance on full training set')
    pp.savefig(plot16)
    plt.close()

    plot17 = plt.figure()
    plt.plot(x_range, val_mcc_mean[0:max_epoch], color='purple')
    plt.fill_between(x_range, val_mcc_mean[0:max_epoch] - val_mcc_error[0:max_epoch], val_mcc_mean[0:max_epoch] + val_mcc_error[0:max_epoch], alpha=0.3, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - Average validation performance')
    pp.savefig(plot17)
    plt.close()

    plot18 = plt.figure()
    plt.plot(zoom_range, val_mcc_mean[start:max_epoch], color='purple')
    plt.fill_between(zoom_range, val_mcc_mean[start:max_epoch] - val_mcc_error[start:max_epoch], val_mcc_mean[start:max_epoch] + val_mcc_error[start:max_epoch], alpha=0.3, color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - Average validation performance')
    pp.savefig(plot18)
    plt.close()

    plot19 = plt.figure()
    plt.plot(x_range, mix_mcc_mean[0:max_epoch], color='green')
    plt.fill_between(x_range, mix_mcc_mean[0:max_epoch] - mix_mcc_error[0:max_epoch], mix_mcc_mean[0:max_epoch] + mix_mcc_error[0:max_epoch], alpha=0.3, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Mixing matrix - Average performance')
    pp.savefig(plot19)
    plt.close()

    plot20 = plt.figure()
    plt.plot(zoom_range, mix_mcc_mean[start:max_epoch], color='green')
    plt.fill_between(zoom_range, mix_mcc_mean[start:max_epoch] - mix_mcc_error[start:max_epoch], mix_mcc_mean[start:max_epoch] + mix_mcc_error[start:max_epoch], alpha=0.3, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('MCS')
    plt.title('Mixing matrix - Average performance (linear assignment)')
    pp.savefig(plot20)
    plt.close()
    
    plot20b = plt.figure()
    plt.plot(zoom_range, mix_mcs_mean[start:max_epoch], color='green')
    plt.fill_between(zoom_range, mix_mcs_mean[start:max_epoch] - mix_mcs_error[start:max_epoch], mix_mcs_mean[start:max_epoch] + mix_mcs_error[start:max_epoch], alpha=0.3, color='green')
    plt.xlabel('Epoch')
    plt.ylabel('MCS')
    plt.title('Mixing matrix - Average performance (permutations)')
    pp.savefig(plot20b)
    plt.close()

    plot21 = plt.figure()
    plt.plot(x_range, all_mcc_mean[0:max_epoch], label='Training', color='pink')
    plt.fill_between(x_range, all_mcc_mean[0:max_epoch] - all_mcc_error[0:max_epoch], all_mcc_mean[0:max_epoch] + all_mcc_error[0:max_epoch], alpha=0.3, color='pink')
    plt.plot(x_range, val_mcc_mean[0:max_epoch], label='Validation', color='purple')
    plt.fill_between(x_range, val_mcc_mean[0:max_epoch] - val_mcc_error[0:max_epoch], val_mcc_mean[0:max_epoch] + val_mcc_error[0:max_epoch], alpha=0.3, color='purple')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - Average performance')
    pp.savefig(plot21)
    plt.close()

    plot22 = plt.figure()
    # Zoom in
    plt.plot(zoom_range, all_mcc_mean[start:max_epoch], label='Training', color='pink')
    plt.fill_between(zoom_range, all_mcc_mean[start:max_epoch] - all_mcc_error[start:max_epoch], all_mcc_mean[start:max_epoch] + all_mcc_error[start:max_epoch], alpha=0.3, color='pink')
    plt.plot(zoom_range, val_mcc_mean[start:max_epoch], label='Validation', color='purple')
    plt.fill_between(zoom_range, val_mcc_mean[start:max_epoch] - val_mcc_error[start:max_epoch], val_mcc_mean[start:max_epoch] + val_mcc_error[start:max_epoch], alpha=0.3, color='purple')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - Average performance')
    pp.savefig(plot22)
    plt.close()

    #### Final plot ####
    f1, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(8,16))

    # Loss plot
    ax1.plot(zoom_range, loss_mean[start:max_epoch], label='Training', color='pink')
    ax1.fill_between(zoom_range, loss_mean[start:max_epoch] - loss_error[start:max_epoch], loss_mean[start:max_epoch] + loss_error[start:max_epoch], alpha=0.3, color='pink')

    ax1.plot(zoom_range, val_loss_mean[start:max_epoch], label='Validation', color='purple')
    ax1.fill_between(zoom_range, val_loss_mean[start:max_epoch] - val_loss_error[start:max_epoch], val_loss_mean[start:max_epoch] + val_loss_error[start:max_epoch], alpha=0.3, color='purple')

    # ax1.legend(locolor='right')
    ax1.set_xlabel('Epoch', size=12)
    ax1.set_ylabel('-ELBO', size=12)
    ax1.set_title('Average Loss', size=12)

    # MCC sources plot
    ax2.plot(zoom_range, all_mcc_mean[start:max_epoch], label='Training', color='pink')
    ax2.fill_between(zoom_range, all_mcc_mean[start:max_epoch] - all_mcc_error[start:max_epoch], all_mcc_mean[start:max_epoch] + all_mcc_error[start:max_epoch], alpha=0.3, color='pink')

    ax2.plot(zoom_range, val_mcc_mean[start:max_epoch], label='Validation', color='purple')
    ax2.fill_between(zoom_range, val_mcc_mean[start:max_epoch] - val_mcc_error[start:max_epoch], val_mcc_mean[start:max_epoch] + val_mcc_error[start:max_epoch], alpha=0.3, color='purple')

    ax2.set_xlabel('Epoch', size=12)
    ax2.set_ylabel('MCC', size=12)
    ax2.set_title('Sources - Average performance', size=12)

    ax2.legend(fontsize=12)

    # MCS mixing matrix (linear assignment) plot
    ax3.plot(zoom_range, mix_mcc_mean[start:max_epoch], color='green')
    ax3.fill_between(zoom_range, mix_mcc_mean[start:max_epoch] - mix_mcc_error[start:max_epoch], mix_mcc_mean[start:max_epoch] + mix_mcc_error[start:max_epoch], alpha=0.3, color='green')

    ax3.set_xlabel('Epoch', size=12)
    ax3.set_ylabel('MCS', size=12)
    ax3.set_title('Mixing matrix - Average performance (linear assignment)', size=12)
    
    # MCS mixing matrix (permutations) plot
    ax4.plot(zoom_range, mix_mcs_mean[start:max_epoch], color='green')
    ax4.fill_between(zoom_range, mix_mcs_mean[start:max_epoch] - mix_mcs_error[start:max_epoch], mix_mcs_mean[start:max_epoch] + mix_mcs_error[start:max_epoch], alpha=0.3, color='green')

    ax4.set_xlabel('Epoch', size=12)
    ax4.set_ylabel('MCS', size=12)
    ax4.set_title('Mixing matrix - Average performance (permutations)', size=12)

    pp.savefig(f1)
    plt.close()
    ####-####

    pp.close()
    
    
def plot_individual(run, config, s, seed, ckpt_freq, exp_id, pdf_name='ind_plots.pdf', start=10):
    '''
    run: (str) path where the run is logged ('run/' by default)
    s: (int) data seed
    seed: (int) learning seed
    ckpt_freq: (int) checkpoint frequency
    exp_id: (int) experiment ID
    pdf_name: (str) path to save the pdf containing the plots.
    start: (int) start epoch for plots
    '''
    # checkpoint frequency
    check_freq = 1 # otherwise, not implemented!!!
#     check_freq = ckpt_freq
    
    file_path = str(run) + '/' + os.path.splitext(config)[0] + '_ds_' + str(s) + '_ls_' + str(seed) + '_' + str(exp_id)  + '.csv'
    df = pd.read_csv(file_path)
        
    n_epochs = len(df['MCC'])
    max_epoch = n_epochs
    
    pp = PdfPages(pdf_name)

    ## Individual learning seeds

    plot1 = plt.figure()
    max_i = len(df['Loss'])
    zoom_i = np.arange(start, max_i)
    plt.plot(zoom_i, df['Loss'].values[start:])
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Training batch loss')
    pp.savefig(plot1)
    plt.close()

    plot2 = plt.figure()
    plt.plot(zoom_i, df['Val_Loss'].values[start:])
    plt.xlabel('Epoch')
    plt.ylabel('-ELBO')
    plt.title('Validation loss')
    pp.savefig(plot2)
    plt.close()

    plot4 = plt.figure()
    x_range_i = np.arange(1, max_i+1)
    x_range_i = x_range_i[x_range_i % check_freq == 0]
    plt.plot(x_range_i, df['MCC_batch'].values)
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - MCC - Training batch')
    pp.savefig(plot4)
    plt.close()

    plot5 = plt.figure()
    plt.plot(x_range_i, df['MCC'].values)
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - MCC training set')
    pp.savefig(plot5)
    plt.close()

    plot6 = plt.figure()
    plt.plot(x_range_i, df['Val_Loss'].values)
    plt.xlabel('Epoch')
    plt.ylabel('MCC')
    plt.title('Sources - MCC validation set')
    pp.savefig(plot6)
    plt.close()

    plot7 = plt.figure()
    plt.plot(x_range_i, df['MCS'].values)
    plt.xlabel('Epoch')
    plt.ylabel('MCS')
    plt.title('Mixing matrix MCS (linear assignment)')
    pp.savefig(plot7)
    plt.close()
    
    plot7b = plt.figure()
    plt.plot(x_range_i, df['MCS_p'].values)
    plt.xlabel('Epoch')
    plt.ylabel('MCS')
    plt.title('Mixing matrix MCS (permutations)')
    pp.savefig(plot7b)
    plt.close()
    
    plot7c = plt.figure()
    plt.plot(x_range_i, df['Grad'].values)
    plt.xlabel('Epoch')
    plt.ylabel(r'$\vert \vert \nabla \vert \vert$')
    plt.title('Norm of the gradients')
    pp.savefig(plot7c)
    plt.close()
    
    plot7d = plt.figure()
    plt.plot(x_range_i, df['Param_diff'].values)
    plt.xlabel('Epoch')
    plt.ylabel(r'$\vert \vert \theta_i - \theta_{i-1} \vert \vert$')
    plt.title('Norm of the difference between current and previous parameters')
    pp.savefig(plot7d)
    plt.close()

    pp.close()