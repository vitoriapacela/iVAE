import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
import yaml
import glob
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('__file__'), '../')))

from iVAE.models.nets import u_simple_IVAE, simple_IVAE, discrete_simple_IVAE
from iVAE.metrics.mcc import mean_corr_coef as mcc
from iVAE.metrics.mcc import max_perm_mcs_col as mcs_p
from matplotlib.backends.backend_pdf import PdfPages

import seaborn as sns; sns.set(style="ticks", color_codes=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

torch.set_default_dtype(torch.double)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)

    
def continuous(config, ckpt, dset, m_ckpt, pdf_name='weight.pdf', n_points=10000):
    '''
    Plots - Continuous case. Same data seed and last learning seed.
    Save to pdf.

    config: (dict) config file
    ckpt: (str) checkpoint path, last epoch
    dset: (SyntheticDataset(Dataset)) test set
    m_ckpt: (str) checkpoint path, all the weights
    pdf_name: (str) path/name of the pdf file where to save the plots.
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    pp = PdfPages(pdf_name)

    # sort by epoch
    f_i = glob.glob(m_ckpt)
#     f_i.sort(key=os.path.getmtime)
    f_i.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    d_data, d_latent, d_aux = dset.get_dims()

    if config.uncentered:
        model = u_simple_IVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim, n_layers=config.n_layers, activation=config.activation, slope=.1, simple_g=True, simple_prior=config.simple_prior, simple_logv=config.simple_logv, noise_level=config.noisy).to(config.device)
    else:
        model = simple_IVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim, n_layers=config.n_layers, activation=config.activation, slope=.1, simple_g=True, simple_prior=config.simple_prior, simple_logv=config.simple_logv, noise_level=config.noisy).to(config.device)

    model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
    Xt, Ut = dset.x.to(device), dset.u.to(device)
    if config.uncentered:
        f, g, v, s, m, l = model(Xt.double(), Ut.double())
    else:
        f, g, v, s, l = model(Xt.double(), Ut.double())
    params = {'decoder': f, 'encoder': g, 'prior': l}
    one_hot_decoded = np.argmax(dset.u, axis=1)

    f0 = plt.figure()
    f0.clf()
    f0.text(0.5,0.8, 'MCS of mixing matrix (linear assignment):', transform=f0.transFigure, size=20, ha="center")
    f0.text(0.5,0.6, mcc(model.f.fc[0].weight.data.detach().cpu().numpy(), dset.A_mix, method='cos'), transform=f0.transFigure, size=18, ha="center")
    pp.savefig(f0)
    plt.close()
    
    f0b = plt.figure()
    f0b.clf()
    f0b.text(0.5,0.8, 'MCS of mixing matrix (permutations):', transform=f0b.transFigure, size=20, ha="center")
    f0b.text(0.5,0.6, mcs_p(model.f.fc[0].weight.data.detach().cpu().numpy(), dset.A_mix, method='cos'), transform=f0b.transFigure, size=18, ha="center")
    pp.savefig(f0b)
    plt.close()

    if config.dd == 2 and config.dl == 2:
        # could be extended to more components, but it would be difficult to see
        fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
        ax1 = plt.subplot(1,3,1)
        sc1 = ax1.scatter(dset.s[:,0], dset.s[:,1], alpha=0.5, c=one_hot_decoded, label=one_hot_decoded)
        legend1 = ax1.legend(*sc1.legend_elements(), title="Segment")
        ax1.add_artist(legend1)
        ax1.set_xlabel(r'$\mathbf{s}_1$')
        ax1.set_ylabel(r'$\mathbf{s}_2$')
        ax1.set_title('Sources')
        ax2 = plt.subplot(1,3,2)
        sc2 = ax2.scatter(Xt.cpu().detach().numpy()[:,0], Xt.cpu().detach().numpy()[:,1], alpha=0.5, c=one_hot_decoded, label=one_hot_decoded)
        # legend2 = ax2.legend(*sc2.legend_elements(), title="Segment")
        # ax2.add_artist(legend1)
        ax2.set_xlabel(r'$\mathbf{x}_1$')
        ax2.set_ylabel(r'$\mathbf{x}_2$')
        ax2.set_title('Observations')
        ax3 = plt.subplot(1,3,3)
        sc3 = ax3.scatter(s[:,0].detach().cpu().numpy(), s[:,1].detach().cpu().numpy(), alpha=0.5, c=one_hot_decoded, label=one_hot_decoded)
        ax3.set_xlabel(r'$\mathbf{\hat{z}}_1$')
        ax3.set_ylabel(r'$\mathbf{\hat{z}}_2$')
        ax3.set_title(r'iVAE latent variables')
        pp.savefig(fig1)
        plt.close()
        
    # to fix!!
    if config.dd == 5 and config.dl == 5:
        df1 = pd.DataFrame()
        df1['True_z_1'] = dset.s[:,0].detach().cpu().numpy()
        df1['True_z_2'] = dset.s[:,1].detach().cpu().numpy()
        df1['True_z_3'] = dset.s[:,2].detach().cpu().numpy()
        df1['True_z_4'] = dset.s[:,3].detach().cpu().numpy()
        df1['True_z_5'] = dset.s[:,4].detach().cpu().numpy()
        df1['Est_z_1'] = s[:,0].detach().cpu().numpy()
        df1['Est_z_2'] = s[:,1].detach().cpu().numpy()
        df1['Est_z_3'] = s[:,2].detach().cpu().numpy()
        df1['Est_z_4'] = s[:,3].detach().cpu().numpy()
        df1['Est_z_5'] = s[:,4].detach().cpu().numpy()

        g1 = sns.pairplot(df1, x_vars=['True_z_1', 'True_z_2', 'True_z_3', 'True_z_4', 'True_z_5'],
                y_vars = ['Est_z_1', 'Est_z_2', 'Est_z_3', 'Est_z_4', 'Est_z_5'])
        g1.fig.suptitle('Estimated sources VS True sources', y=1.04)
        pp.savefig(g1.fig)
        plt.close()

    if config.dd == 2 and config.dl == 2:
        # could be extended to more dimensions, but it would be difficult to see
        fig2, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(18, 5), sharex=True, sharey=True)
        ax1 = plt.subplot(1,3,1)
        ax1.plot(dset.s[:,0][0:n_points], label=r"$\mathbf{s}_1$", alpha=0.6)
        ax1.plot(dset.s[:,1][0:n_points], label=r"$\mathbf{s}_2$", alpha=0.6)
        ax1.set_title('Sources')
        ax1.legend()
        ax2 = plt.subplot(1,3,2)
        ax2.plot(Xt[:,0].cpu().detach().numpy()[0:n_points], label=r"$\mathbf{x}_1$", alpha=0.6)
        ax2.plot(Xt[:,1].cpu().detach().numpy()[0:n_points], label=r"$\mathbf{x}_2$", alpha=0.6)
        ax2.set_title('Observations')
        ax2.legend()
        ax3 = plt.subplot(1,3,3)
        ax3.plot(s[:,0].cpu().detach().numpy()[0:n_points], label=r"$\mathbf{\hat{z}}_1$", alpha=0.6)
        ax3.plot(s[:,1].cpu().detach().numpy()[0:n_points], label=r"$\mathbf{\hat{z}}_2$", alpha=0.6)
        ax3.set_title('iVAE latent variables')
        ax3.legend()
        ax2.set_xlabel('Data point')
        pp.savefig(fig2)
        plt.close()

#     f3 = plt.figure()
#     plt.scatter(dset.s[:,0][0:n_points], s[:,0].cpu().detach().numpy()[0:n_points], alpha=0.1, s=10)
#     plt.xlabel('True source')
#     plt.ylabel('Estimated source')
#     plt.title('True component 1 X  estimated component 1')
#     pp.savefig(f3)
#     plt.close()

#     f4 = plt.figure()
#     plt.scatter(dset.s[:,0][0:n_points], s[:,1].cpu().detach().numpy()[0:n_points], alpha=0.1, s=10)
#     plt.xlabel('True source')
#     plt.ylabel('Estimated source')
#     plt.title('True component 1 X estimated component 2')
#     pp.savefig(f4)
#     plt.close()

#     f5 = plt.figure()
#     plt.scatter(dset.s[:,1][0:n_points], s[:,0].cpu().detach().numpy()[0:n_points], alpha=0.1, s=10)
#     plt.xlabel('True source')
#     plt.ylabel('Estimated source')
#     plt.title('True component 2 X estimated component 1')
#     pp.savefig(f5)
#     plt.close()

#     f6 = plt.figure()
#     plt.scatter(dset.s[:,1][0:n_points], s[:,1].cpu().detach().numpy()[0:n_points], alpha=0.1, s=10)
#     plt.xlabel('True source')
#     plt.ylabel('Estimated source')
#     plt.title('True component 2 X estimated component 2')
#     pp.savefig(f6)
#     plt.close()

    # if the matrix is too big, it won't fit
    if config.dd < 6:
        f7 = plt.figure(figsize=(10,10))
        f7.clf()
        txt1 = 'True mixing matrix:'
        f7.text(0.5,0.8, txt1, transform=f7.transFigure, size=20, ha="center")
        txt2 = dset.A_mix
        f7.text(0.5,0.6, txt2, transform=f7.transFigure, size=20, ha="center")
        txt3 = 'Estimated mixing matrix:'
        f7.text(0.5,0.4, txt3, transform=f7.transFigure, size=20, ha="center")
        txt4 = model.f.fc[0].weight.data
        f7.text(0.5,0.2, str(txt4), transform=f7.transFigure, size=20, ha="center")
        pp.savefig(f7)
        plt.close()

    f8 = plt.figure()
    f8.clf()
    f8.text(0.5,0.8, 'Eigenvalues:', transform=f8.transFigure, size=20, ha="center")
    f8.text(0.5,0.6, np.linalg.eig(dset.A_mix)[0], transform=f8.transFigure, size=12, ha="center")
    f8.text(0.5,0.4, 'Condition number:', transform=f8.transFigure, size=20, ha="center")
    f8.text(0.5,0.2, np.linalg.cond(dset.A_mix), transform=f8.transFigure, size=20, ha="center")
    pp.savefig(f8)
    plt.close()

    f9 = plt.figure()
    plt.title('Estimated observations')
    plt.plot(f[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label=('$\mathbf{\hat{x}}_1$'))
    plt.plot(f[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label=('$\mathbf{\hat{x}}_2$'))
    plt.xlabel('Data point')
    plt.ylabel('$f(\mathbf{z})$')
    plt.legend()
    pp.savefig(f9)
    plt.close()

    f10 = plt.figure()
    plt.title('Mean: inference model')
    plt.plot(g[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{\mu}}_1$')
    plt.plot(g[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{\mu}}_2$')
    plt.xlabel('Data point')
    plt.ylabel('$g(\mathbf{x,u})$')
    plt.legend()
    pp.savefig(f10)
    plt.close()

    f11 = plt.figure()
    plt.title('Variance: inference model')
    plt.plot(v[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{\sigma}}_1$')
    plt.plot(v[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{\sigma}}_2$')
    plt.xlabel('Data point')
    plt.ylabel('$V(\mathbf{x,u})$')
    plt.legend()
    pp.savefig(f11)
    plt.close()

    f12 = plt.figure()
    plt.title('Estimated sources')
    plt.plot(s[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{s}}_1$')
    plt.plot(s[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{s}}_2$')
    plt.xlabel('Data point')
    plt.ylabel('$\mathbf{s}$')
    plt.legend()
    pp.savefig(f12)
    plt.close()

    f13 = plt.figure()
    plt.title('Estimated prior variance')
    plt.plot(l[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{l}}_1$')
    plt.plot(l[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{l}}_2$')
    plt.xlabel('Data point')
    plt.ylabel('$\lambda(\mathbf{u})$')
    plt.legend()
    pp.savefig(f13)
    plt.close()

    f14 = plt.figure()
    plt.plot(dset.l[:,0][0:n_points], label='$\mathbf{l}_1$')
    plt.plot(dset.l[:,1][0:n_points], label='$\mathbf{l}_2$')
    plt.title('True prior variance')
    plt.xlabel('Segment')
    plt.ylabel('$\lambda(\mathbf{u})$')
    plt.legend()
    pp.savefig(f14)
    plt.close()

#     # True prior variance per data point
#     L_var = np.zeros((len(dset.x), config.dl))
#     for seg in range(len(dset.l)):
#         segID = range(config.nps * seg, config.nps * (seg + 1))
#         L_var[segID] = dset.l[seg]

#     f15 = plt.figure()
#     plt.plot(L_var[:,0][0:n_points], label='$\mathbf{l}_1$')
#     plt.plot(L_var[:,1][0:n_points], label='$\mathbf{l}_2$')
#     plt.title('True prior variances')
#     plt.xlabel('Data point')
#     plt.ylabel('$\lambda(\mathbf{u})$')
#     plt.legend()
#     pp.savefig(f15)
#     plt.close()
    
    f15 = plt.figure()
    plt.plot(dset.l[:,0], label='$\mathbf{l}_1$')
    plt.plot(dset.l[:,1], label='$\mathbf{l}_2$')
    plt.title('True prior variances')
    plt.xlabel('Segment')
    plt.ylabel('$\lambda(\mathbf{u})$')
    plt.legend()
    pp.savefig(f15)
    plt.close()
    
    # Do not use log(L_var) if L_var = 1
    if not config.same_var:
#         f20 = plt.figure()
#         f20.clf()
#         f20.text(0.5,0.8, 'MCC of prior variance:', transform=f20.transFigure, size=20, ha="center")
#         f20.text(0.5,0.6, mcc(L_var, l.cpu().detach().numpy()), transform=f20.transFigure, size=18, ha="center")
#         pp.savefig(f20)
#         plt.close()

        est_var = (model.logl.fc[0].weight.cpu().detach().numpy()).T
    
        f20 = plt.figure()
        f20.clf()
        f20.text(0.5,0.8, 'MCC of prior variance:', transform=f20.transFigure, size=20, ha="center")
        f20.text(0.5,0.6, mcc(dset.l, est_var), transform=f20.transFigure, size=18, ha="center")
        pp.savefig(f20)
        plt.close()

#     if config.dd == 2 and config.dl == 2:
#         f16 = plt.figure()
#         plt.plot(np.log(L_var[:,0][0:n_points]**2), label='True')
#         plt.plot(np.log(l[:,0].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#         plt.title('Prior variance: source 1 x component 1')
#         plt.xlabel('Data point')
#         plt.ylabel('$\log \lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f16)
#         plt.close()

#         f17 = plt.figure()
#         plt.plot(np.log(L_var[:,0][0:n_points]**2), label='True')
#         plt.plot(np.log(l[:,1].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#         plt.title('Prior variance: source 1 x component 2')
#         plt.xlabel('Data point')
#         plt.ylabel('$\log \lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f17)
#         plt.close()

#         f18 = plt.figure()
#         plt.plot(np.log(L_var[:,1][0:n_points]**2), label='True')
#         plt.plot(np.log(l[:,0].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#         plt.title('Prior variance: source 2 x component 1')
#         plt.xlabel('Data point')
#         plt.ylabel('$\log \lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f18)
#         plt.close()

#         f19 = plt.figure()
#         plt.plot(np.log(L_var[:,1][0:n_points]**2), label='True')
#         plt.plot(np.log(l[:,1].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#         plt.title('Prior variance: source 2 x component 2')
#         plt.xlabel('Data point')
#         plt.ylabel('$\log \lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f19)
#         plt.close()

    if config.uncentered:
        f21 = plt.figure()
        plt.plot(dset.m[:,0], label='$\mathbf{m}_1$')
        plt.plot(dset.m[:,1], label='$\mathbf{m}_2$')
        plt.title('True prior mean')
        plt.xlabel('Segment')
        plt.ylabel('$\mu(\mathbf{u})$')
        plt.legend()
        pp.savefig(f21)
        plt.close()

#         # True prior mean per data point
#         m_seg = np.zeros((len(dset.x), config.dd))
#         for seg in range(len(dset.m)):
#             segID = range(config.nps * seg, config.nps * (seg + 1))
#             m_seg[segID] = dset.m[seg]

#         f22 = plt.figure()
#         plt.plot(m_seg[:,0][0:n_points], label='$\mathbf{m}_1$')
#         plt.plot(m_seg[:,1][0:n_points], label='$\mathbf{m}_2$')
#         plt.title('True prior means')
#         plt.xlabel('Data point')
#         plt.ylabel('$\lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f22)
#         plt.close()
        
#         f27 = plt.figure()
#         f27.clf()
#         f27.text(0.5,0.8, 'MCC of prior mean:', transform=f27.transFigure, size=20, ha="center")
#         f27.text(0.5,0.6, mcc(m_seg, m.cpu().detach().numpy()), transform=f27.transFigure, size=18, ha="center")
#         pp.savefig(f27)
#         plt.close()
        
        est_mean = (model.prior_mean.fc[0].weight.cpu().detach().numpy()).T
        
        f22 = plt.figure()
        plt.plot(est_mean[:,0], label='$\mathbf{m}_1$')
        plt.plot(est_mean[:,1], label='$\mathbf{m}_2$')
        plt.title('Estimated prior means')
        plt.xlabel('Data point')
        plt.ylabel('$\lambda(\mathbf{u})$')
        plt.legend()
        pp.savefig(f22)
        plt.close()
        
        f27 = plt.figure()
        f27.clf()
        f27.text(0.5,0.8, 'MCC of prior mean:', transform=f27.transFigure, size=20, ha="center")
        f27.text(0.5,0.6, mcc(dset.m, est_mean), transform=f27.transFigure, size=18, ha="center")
        pp.savefig(f27)
        plt.close()

#         if config.dd == 2 and config.dl == 2:
#             f23 = plt.figure()
#             plt.plot(np.log(m_seg[:,0][0:n_points]**2), label='True')
#             plt.plot(np.log(m[:,0].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#             plt.title('Prior mean: source 1 x component 1')
#             plt.xlabel('Data point')
#             plt.ylabel('$\log \mu(\mathbf{u})$')
#             plt.legend()
#             pp.savefig(f23)
#             plt.close()

#             f24 = plt.figure()
#             plt.plot(np.log(m_seg[:,0][0:n_points]**2), label='True')
#             plt.plot(np.log(m[:,1].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#             plt.title('Prior mean: source 1 x component 2')
#             plt.xlabel('Data point')
#             plt.ylabel('$\log \mu(\mathbf{u})$')
#             plt.legend()
#             pp.savefig(f24)
#             plt.close()

#             f25 = plt.figure()
#             plt.plot(np.log(m_seg[:,1][0:n_points]**2), label='True')
#             plt.plot(np.log(m[:,0].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#             plt.title('Prior mean: source 2 x component 1')
#             plt.xlabel('Data point')
#             plt.ylabel('$\log \mu(\mathbf{u})$')
#             plt.legend()
#             pp.savefig(f25)
#             plt.close()

#             f26 = plt.figure()
#             plt.plot(np.log(m_seg[:,1][0:n_points]**2), label='True')
#             plt.plot(np.log(m[:,1].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#             plt.title('Prior mean: source 2 x component 2')
#             plt.xlabel('Data point')
#             plt.ylabel('$\log \mu(\mathbf{u})$')
#             plt.legend()
#             pp.savefig(f26)
#             plt.close()

    # mixing matrix weight per epoch
#     max_epoch = config.epochs
    max_epoch = int(ckpt.split('/')[-1].split('.')[0].split('_')[-1])
    x_range = np.arange(1, max_epoch+1)
    fs = []
    for ckpt in f_i:
        m = torch.load(ckpt, map_location=device)['model_state_dict']
        fs.append(m['f.fc.0.weight'].cpu().detach().numpy())
    fs = np.asarray(fs)
    fs = fs[0:max_epoch]
    # labels ordered column-wise
    ff = plt.figure()
    plt.plot(x_range, fs[:, 0, 0], label="$f_{11}$", alpha=0.5, marker='o')
    plt.plot(x_range, fs[:, 1, 0], label="$f_{21}$", alpha=0.5, marker='o')
    plt.plot(x_range, fs[:, 0, 1], label="$f_{12}$", alpha=0.5, marker='o')
    plt.plot(x_range, fs[:, 1, 1], label="$f_{22}$", alpha=0.5, marker='o')
    plt.legend(loc='right')
    plt.xlabel('Epoch')
    plt.title('Weights of f (decoder/mixing model)')
    pp.savefig(ff)
    plt.close()

    pp.close()


def disc_plots(config, dset, m_ckpt, pdf_name='weight.pdf', n_points=10000, check_freq=100):
    '''
    Plots - Discrete case. Same data seed and last learning seed.
    Save to pdf.

    config: (dict) config file
    dset: (SyntheticDataset(Dataset)) test set
    m_ckpt: (str) checkpoint path, all the weights
    pdf_name: (str) path/name of the pdf file where to save the plots.
    check_freq: (int) frequency in which the checkpoint files were saved, e.g. every 100 epochs.
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    pp = PdfPages(pdf_name)

    # sort by epoch
    f_i = glob.glob(m_ckpt)
#     f_i.sort(key=os.path.getmtime)
    f_i.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
#     f_i = f_i[0::sample_freq]
    
    max_epoch = int(f_i[-1].split('/')[-1].split('.')[0].split('_')[-1])
    x_range = np.arange(1, max_epoch+1)
#     x_range = x_range[0::sample_freq]
    x_range = x_range[0::check_freq]
    
    ckpt = f_i[-1] # path to checkpoint of last epoch

    d_data, d_latent, d_aux = dset.get_dims()

    model = discrete_simple_IVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim, n_layers=config.n_layers, activation=config.activation, slope=.1, simple_g=True, simple_prior=config.simple_prior, simple_logv=config.simple_logv, fix_v=config.fix_v, logvar=config.logvar).to(config.device)

    model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
    Xt, Ut = dset.x.to(device), dset.u.to(device)
    f, g, v, s, m, l = model(Xt.double(), Ut.double())
    params = {'decoder': f, 'encoder': g, 'prior': l}
    one_hot_decoded = np.argmax(dset.u, axis=1)

    f0 = plt.figure()
    f0.clf()
    f0.text(0.5,0.8, 'MCS of mixing matrix (linear assignment):', transform=f0.transFigure, size=20, ha="center")
    f0.text(0.5,0.6, mcc(model.f.fc[0].weight.data.detach().cpu().numpy(), dset.A_mix, method='cos'), transform=f0.transFigure, size=18, ha="center")
    pp.savefig(f0)
    plt.close()
    
    f0b = plt.figure()
    f0b.clf()
    f0b.text(0.5,0.8, 'MCS of mixing matrix (permutations):', transform=f0b.transFigure, size=20, ha="center")
    f0b.text(0.5,0.6, mcs_p(model.f.fc[0].weight.data.detach().cpu().numpy(), dset.A_mix, method='cos'), transform=f0b.transFigure, size=18, ha="center")
    pp.savefig(f0b)
    plt.close()

    # to do: pairs plots more components
    f3 = plt.figure()
    plt.scatter(dset.s[:,0][0:n_points], s[:,0].cpu().detach().numpy()[0:n_points], alpha=0.1, s=10)
    plt.xlabel('True source')
    plt.ylabel('Estimated source')
    plt.title('True component 1 X  estimated component 1')
    pp.savefig(f3)
    plt.close()

    if (config.dl > 1) and (config.dd > 1):
        f4 = plt.figure()
        plt.scatter(dset.s[:,0][0:n_points], s[:,1].cpu().detach().numpy()[0:n_points], alpha=0.1, s=10)
        plt.xlabel('True source')
        plt.ylabel('Estimated source')
        plt.title('True component 1 X estimated component 2')
        pp.savefig(f4)
        plt.close()

        f5 = plt.figure()
        plt.scatter(dset.s[:,1][0:n_points], s[:,0].cpu().detach().numpy()[0:n_points], alpha=0.1, s=10)
        plt.xlabel('True source')
        plt.ylabel('Estimated source')
        plt.title('True component 2 X estimated component 1')
        pp.savefig(f5)
        plt.close()

        f6 = plt.figure()
        plt.scatter(dset.s[:,1][0:n_points], s[:,1].cpu().detach().numpy()[0:n_points], alpha=0.1, s=10)
        plt.xlabel('True source')
        plt.ylabel('Estimated source')
        plt.title('True component 2 X estimated component 2')
        pp.savefig(f6)
        plt.close()

    # if the matrix is too big, it won't fit
    if (config.dd == 2) and (config.dl == 2):
        f7 = plt.figure()
        f7.clf()
        txt1 = 'True mixing matrix:'
        f7.text(0.5,0.8, txt1, transform=f7.transFigure, size=20, ha="center")
        txt2 = dset.A_mix
        f7.text(0.5,0.6, txt2, transform=f7.transFigure, size=20, ha="center")
        txt3 = 'Estimated mixing matrix:'
        f7.text(0.5,0.4, txt3, transform=f7.transFigure, size=20, ha="center")
        txt4 = model.f.fc[0].weight.data
        f7.text(0.5,0.2, str(txt4), transform=f7.transFigure, size=20, ha="center")
        pp.savefig(f7)
        plt.close()

    if (config.dl > 1) and (config.dd > 1) and (config.dl == config.dd):
        f8 = plt.figure()
        f8.clf()
        f8.text(0.5,0.8, 'Eigenvalues:', transform=f8.transFigure, size=20, ha="center")
        f8.text(0.5,0.6, np.linalg.eig(dset.A_mix)[0], transform=f8.transFigure, size=12, ha="center")
        f8.text(0.5,0.4, 'Condition number:', transform=f8.transFigure, size=20, ha="center")
        f8.text(0.5,0.2, np.linalg.cond(dset.A_mix), transform=f8.transFigure, size=20, ha="center")
        pp.savefig(f8)
        plt.close()

    if (config.dl > 1) and (config.dd > 1):
        f9 = plt.figure()
        plt.title('Estimated observations')
        plt.plot(f[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label=('$\mathbf{\hat{x}}_1$'))
        plt.plot(f[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label=('$\mathbf{\hat{x}}_2$'))
        plt.xlabel('Data point')
        plt.ylabel('$f(\mathbf{z})$')
        plt.legend()
        pp.savefig(f9)
        plt.close()

        f10 = plt.figure()
        plt.title('Mean: inference model')
        plt.plot(g[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{\mu}}_1$')
        plt.plot(g[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{\mu}}_2$')
        plt.xlabel('Data point')
        plt.ylabel('$g(\mathbf{x,u})$')
        plt.legend()
        pp.savefig(f10)
        plt.close()

        if not config.fix_v:
            f11 = plt.figure()
            plt.title('Variance: inference model')
            plt.plot(v[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{\sigma}}_1$')
            plt.plot(v[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{\sigma}}_2$')
            plt.xlabel('Data point')
            plt.ylabel('$V(\mathbf{x,u})$')
            plt.legend()
            pp.savefig(f11)
            plt.close()

        f12 = plt.figure()
        plt.title('Estimated sources')
        plt.plot(s[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{s}}_1$')
        plt.plot(s[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{s}}_2$')
        plt.xlabel('Data point')
        plt.ylabel('$\mathbf{s}$')
        plt.legend()
        pp.savefig(f12)
        plt.close()

        f13 = plt.figure()
        plt.title('Estimated prior variance')
        plt.plot(l[:,0].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{l}}_1$')
        plt.plot(l[:,1].cpu().detach().numpy()[0:n_points], alpha=0.6, label='$\mathbf{\hat{l}}_2$')
        plt.xlabel('Data point')
        plt.ylabel('$\lambda(\mathbf{u})$')
        plt.legend()
        pp.savefig(f13)
        plt.close()

        f14 = plt.figure()
        plt.plot(dset.l[:,0][0:n_points], label='$\mathbf{l}_1$')
        plt.plot(dset.l[:,1][0:n_points], label='$\mathbf{l}_2$')
        plt.title('True prior variance')
        plt.xlabel('Segment')
        plt.ylabel('$\lambda(\mathbf{u})$')
        plt.legend()
        pp.savefig(f14)
        plt.close()

#         # True prior variance per data point
#         L_var = np.zeros((len(dset.x), config.dl))
#         for seg in range(len(dset.l)):
#             segID = range(config.nps * seg, config.nps * (seg + 1))
#             L_var[segID] = dset.l[seg]

#         f15 = plt.figure()
#         plt.plot(L_var[:,0][0:n_points], label='$\mathbf{l}_1$')
#         plt.plot(L_var[:,1][0:n_points], label='$\mathbf{l}_2$')
#         plt.title('True prior variances')
#         plt.xlabel('Data point')
#         plt.ylabel('$\lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f15)
#         plt.close()

        f15 = plt.figure()
        plt.plot(dset.l[:,0], label='$\mathbf{l}_1$')
        plt.plot(dset.l[:,1], label='$\mathbf{l}_2$')
        plt.title('True prior variances')
        plt.xlabel('Segment')
        plt.ylabel('$\lambda(\mathbf{u})$')
        plt.legend()
        pp.savefig(f15)
        plt.close()
    
        # Do not use if L_var = 1
        if not config.same_var:
#             f20 = plt.figure()
#             f20.clf()
#             f20.text(0.5,0.8, 'MCC of prior variance:', transform=f20.transFigure, size=20, ha="center")
#             f20.text(0.5,0.6, mcc(L_var, l.cpu().detach().numpy()), transform=f20.transFigure, size=18, ha="center")
#             pp.savefig(f20)
#             plt.close()

            est_var = (model.logl.fc[0].weight.cpu().detach().numpy()).T
            mcs_var = mcc(dset.l, est_var, method='cos')
            f20 = plt.figure()
            f20.clf()
            f20.text(0.5,0.8, 'MCS of prior log-variance:', transform=f20.transFigure, size=20, ha="center")
            f20.text(0.5,0.6, mcs_var, transform=f20.transFigure, size=18, ha="center")
            pp.savefig(f20)
            plt.close()

#     if config.dd == 2 and config.dl == 2:
#         f16 = plt.figure()
#         plt.plot(np.log(L_var[:,0][0:n_points]**2), label='True')
#         plt.plot(np.log(l[:,0][0:n_points].cpu().detach().numpy()), alpha=0.6, label='Estimated')
#         plt.title('Prior variance: source 1 x component 1')
#         plt.xlabel('Data point')
#         plt.ylabel('$\log \lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f16)
#         plt.close()

#         f17 = plt.figure()
#         plt.plot(np.log(L_var[:,0][0:n_points]**2), label='True')
#         plt.plot(np.log(l[:,1][0:n_points].cpu().detach().numpy()), alpha=0.6, label='Estimated')
#         plt.title('Prior variance: source 1 x component 2')
#         plt.xlabel('Data point')
#         plt.ylabel('$\log \lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f17)
#         plt.close()

#         f18 = plt.figure()
#         plt.plot(np.log(L_var[:,1][0:n_points]**2), label='True')
#         plt.plot(np.log(l[:,0].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#         plt.title('Prior variance: source 2 x component 1')
#         plt.xlabel('Data point')
#         plt.ylabel('$\log \lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f18)
#         plt.close()

#         f19 = plt.figure()
#         plt.plot(np.log(L_var[:,1][0:n_points]**2), label='True')
#         plt.plot(np.log(l[:,1].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#         plt.title('Prior variance: source 2 x component 2')
#         plt.xlabel('Data point')
#         plt.ylabel('$\log \lambda(\mathbf{u})$')
#         plt.legend()
#         pp.savefig(f19)
#         plt.close()

    if (config.dl > 1) and (config.dd > 1):
        if config.uncentered:
            f21 = plt.figure()
            plt.plot(dset.m[:,0][0:n_points], label='$\mathbf{m}_1$')
            plt.plot(dset.m[:,1][0:n_points], label='$\mathbf{m}_2$')
            plt.title('True prior mean')
            plt.xlabel('Segment')
            plt.ylabel('$\mu(\mathbf{u})$')
            plt.legend()
            pp.savefig(f21)
            plt.close()

#             # True prior mean per data point
#             m_seg = np.zeros((len(dset.x), config.dl))
#             for seg in range(len(dset.m)):
#                 segID = range(config.nps * seg, config.nps * (seg + 1))
#                 m_seg[segID] = dset.m[seg]

#             f22 = plt.figure()
#             plt.plot(m_seg[:,0][0:n_points], label='$\mathbf{m}_1$')
#             plt.plot(m_seg[:,1][0:n_points], label='$\mathbf{m}_2$')
#             plt.title('True prior means')
#             plt.xlabel('Data point')
#             plt.ylabel('$\lambda(\mathbf{u})$')
#             plt.legend()
#             pp.savefig(f22)
#             plt.close()
        
#             f27 = plt.figure()
#             f27.clf()
#             f27.text(0.5,0.8, 'MCC of prior mean:', transform=f27.transFigure, size=20, ha="center")
#             f27.text(0.5,0.6, mcc(m_seg, m.cpu().detach().numpy()), transform=f27.transFigure, size=18, ha="center")
#             pp.savefig(f27)
#             plt.close()
            
            est_mean = (model.prior_mean.fc[0].weight.cpu().detach().numpy()).T
        
            if config.dd == 2 and config.dl == 2:
                f22 = plt.figure()
                plt.plot(est_mean[:,0], label='$\mathbf{m}_1$')
                plt.plot(est_mean[:,1], label='$\mathbf{m}_2$')
                plt.title('Estimated prior means')
                plt.xlabel('Data point')
                plt.ylabel('$\lambda(\mathbf{u})$')
                plt.legend()
                pp.savefig(f22)
                plt.close()

            mcs_mean = mcc(dset.m, est_mean, method='cos')
            f27 = plt.figure()
            f27.clf()
            f27.text(0.5,0.8, 'MCS of prior mean:', transform=f27.transFigure, size=20, ha="center")
            f27.text(0.5,0.6, mcs_mean, transform=f27.transFigure, size=18, ha="center")
            pp.savefig(f27)
            plt.close()

#         if config.dd == 2 and config.dl == 2:
#             f23 = plt.figure()
#             plt.plot(np.log(m_seg[:,0][0:n_points]**2), label='True')
#             plt.plot(np.log(m[:,0].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#             plt.title('Prior mean: source 1 x component 1')
#             plt.xlabel('Data point')
#             plt.ylabel('$\log \mu(\mathbf{u})$')
#             plt.legend()
#             pp.savefig(f23)
#             plt.close()

#             f24 = plt.figure()
#             plt.plot(np.log(m_seg[:,0][0:n_points]**2), label='True')
#             plt.plot(np.log(m[:,1].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#             plt.title('Prior mean: source 1 x component 2')
#             plt.xlabel('Data point')
#             plt.ylabel('$\log \mu(\mathbf{u})$')
#             plt.legend()
#             pp.savefig(f24)
#             plt.close()

#             f25 = plt.figure()
#             plt.plot(np.log(m_seg[:,1][0:n_points]**2), label='True')
#             plt.plot(np.log(m[:,0].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#             plt.title('Prior mean: source 2 x component 1')
#             plt.xlabel('Data point')
#             plt.ylabel('$\log \mu(\mathbf{u})$')
#             plt.legend()
#             pp.savefig(f25)
#             plt.close()

#             f26 = plt.figure()
#             plt.plot(np.log(m_seg[:,1][0:n_points]**2), label='True')
#             plt.plot(np.log(m[:,1].cpu().detach().numpy()[0:n_points]), alpha=0.6, label='Estimated')
#             plt.title('Prior mean: source 2 x component 2')
#             plt.xlabel('Data point')
#             plt.ylabel('$\log \mu(\mathbf{u})$')
#             plt.legend()
#             pp.savefig(f26)
#             plt.close()

    # mixing matrix weights per epoch
    if config.dd > 1 and config.dl > 1:
        fs = []
        for f in f_i:
            m = torch.load(f, map_location=device)['model_state_dict']
            fs.append(m['f.fc.0.weight'].cpu().detach().numpy())
        fs = np.asarray(fs)
        # labels ordered column-wise
        ff = plt.figure()
        ax = plt.subplot(111)
        for i in range(config.dd):
            for j in range(config.dl): 
                ax.plot(x_range, fs[:, i, j], label='f_'+str(i)+','+str(j), alpha=0.6)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="x-small", labelspacing=0.05)
        plt.xlabel('Epoch')
        plt.title('Weights of f (decoder/mixing model)')
        pp.savefig(ff)
        plt.close()
    
        # mixing matrix weights per epoch, in separate plots
        fs = []
        for f in f_i:
            m = torch.load(f, map_location=device)['model_state_dict']
            fs.append(m['f.fc.0.weight'].cpu().detach().numpy())
        fs = np.asarray(fs)
        # labels ordered column-wise
        for i in range(config.dd):
            for j in range(config.dl):
                ff = plt.figure()
                ax = plt.subplot(111)
                ax.plot(x_range, fs[:, i, j], label='f_'+str(i)+','+str(j), alpha=0.6)
                plt.xlabel('Epoch')
                plt.title('Mixing model weight ' + str(i) + ', ' + str(j) )
                pp.savefig(ff)
                plt.close()

    
    # prior logvariance (logl) weights per epoch
    if config.track_prior:
        if config.dd > 1 and config.dl > 1:
            fs = []
            for f in f_i:
                m = torch.load(f, map_location=device)['model_state_dict']
                fs.append(m['logl.fc.0.weight'].cpu().detach().numpy())
            fs = np.asarray(fs)
            # labels ordered column-wise
            # create one plot for each component because there are so many
            for i in range(config.dl):
                for j in range(config.ns): 
                    ff = plt.figure()
                    ax = plt.subplot(111)       
                    ax.plot(x_range, fs[:, i, j], alpha=0.6)

                    plt.xlabel('Epoch')
                    plt.title('Prior log-var (logl) weight, Component ' + str(i) + ', ' + str(j))
                    pp.savefig(ff)
                    plt.close() 
                
    # prior means weights per epoch
    if config.track_prior:
        if config.dd > 1 and config.dl > 1:
            fs = []
            for f in f_i:
                m = torch.load(f, map_location=device)['model_state_dict']
                fs.append(m['prior_mean.fc.0.weight'].cpu().detach().numpy())
            fs = np.asarray(fs)
            # labels ordered column-wise
            # create one plot for each component because there are so many
            for i in range(config.dl):
                for j in range(config.ns): 
                    ff = plt.figure()
                    ax = plt.subplot(111)       
                    ax.plot(x_range, fs[:, i, j], alpha=0.6)

                    plt.xlabel('Epoch')
                    plt.title('Prior mean weight, Component ' + str(i) + ', ' + str(j))
                    pp.savefig(ff)
                    plt.close()

    pp.close()
    

def _disc_plots(config, dset, m_ckpt, pdf_name='weight.pdf', n_points=10000, check_freq=100):
    '''
    Plots - Discrete case. Same data seed and last learning seed.
    Save to pdf.

    config: (dict) config file
    dset: (SyntheticDataset(Dataset)) test set
    m_ckpt: (str) checkpoint path, all the weights
    pdf_name: (str) path/name of the pdf file where to save the plots.
    check_freq: (int) frequency in which the checkpoint files were saved, e.g. every 100 epochs.
    '''
    #  Old implementation: sample_freq: (int) frequency to sample datapoints, e.g. every 100 epochs.
    # Not in use anymore because now we are checkpointing sparsely and that is enough.
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    pp = PdfPages(pdf_name)

    # sort by epoch
    f_i = glob.glob(m_ckpt)
#     f_i.sort(key=os.path.getmtime)
    f_i.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    max_epoch = int(f_i[-1].split('/')[-1].split('.')[0].split('_')[-1])
    x_range = np.arange(1, max_epoch+1)
    x_range = x_range[0::check_freq]
    
    ckpt = f_i[-1] # path to checkpoint of last epoch

    d_data, d_latent, d_aux = dset.get_dims()

    model = discrete_simple_IVAE(data_dim=d_data, latent_dim=d_latent, aux_dim=d_aux, hidden_dim=config.hidden_dim, n_layers=config.n_layers, activation=config.activation, slope=.1, simple_g=True, simple_prior=config.simple_prior, simple_logv=config.simple_logv, fix_v=config.fix_v, logvar=config.logvar).to(config.device)

    model.load_state_dict(torch.load(ckpt, map_location=device)['model_state_dict'])
    Xt, Ut = dset.x.to(device), dset.u.to(device)
    f, g, v, s, m, l = model(Xt.double(), Ut.double())
    params = {'decoder': f, 'encoder': g, 'prior': l}
    one_hot_decoded = np.argmax(dset.u, axis=1)

    f0 = plt.figure()
    f0.clf()
    f0.text(0.5,0.8, 'MCS of mixing matrix (linear assignment):', transform=f0.transFigure, size=20, ha="center")
    f0.text(0.5,0.6, mcc(model.f.fc[0].weight.data.detach().cpu().numpy(), dset.A_mix, method='cos'), transform=f0.transFigure, size=18, ha="center")
    pp.savefig(f0)
    plt.close()
    
    f0b = plt.figure()
    f0b.clf()
    f0b.text(0.5,0.8, 'MCS of mixing matrix (permutations):', transform=f0b.transFigure, size=20, ha="center")
    f0b.text(0.5,0.6, mcs_p(model.f.fc[0].weight.data.detach().cpu().numpy(), dset.A_mix, method='cos'), transform=f0b.transFigure, size=18, ha="center")
    pp.savefig(f0b)
    plt.close()

    if (config.dl == config.dd) and (config.dl > 1):
        f8 = plt.figure()
        f8.clf()
        f8.text(0.5,0.8, 'Eigenvalues:', transform=f8.transFigure, size=20, ha="center")
        f8.text(0.5,0.6, np.linalg.eig(dset.A_mix)[0], transform=f8.transFigure, size=12, ha="center")
        f8.text(0.5,0.4, 'Condition number:', transform=f8.transFigure, size=20, ha="center")
        f8.text(0.5,0.2, np.linalg.cond(dset.A_mix), transform=f8.transFigure, size=20, ha="center")
        pp.savefig(f8)
        plt.close()

    # mixing matrix weights per epoch
    if config.dl > 1 and config.dd > 1:
        fs = []
        for f in f_i:
            m = torch.load(f, map_location=device)['model_state_dict']
            fs.append(m['f.fc.0.weight'].cpu().detach().numpy())
        fs = np.asarray(fs)
        # labels ordered column-wise
        ff = plt.figure()
        ax = plt.subplot(111)
        for i in range(config.dd):
            for j in range(config.dl): 
                ax.plot(x_range, fs[:, i, j], label='f_'+str(i)+','+str(j), alpha=0.6)
        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize="x-small", labelspacing=0.05)
        plt.xlabel('Epoch')
        plt.title('Weights of f (decoder/mixing model)')
        pp.savefig(ff)
        plt.close()

        # mixing matrix weights per epoch, in separate plots
        fs = []
        for f in f_i:
            m = torch.load(f, map_location=device)['model_state_dict']
            fs.append(m['f.fc.0.weight'].cpu().detach().numpy())
        fs = np.asarray(fs)
        # labels ordered column-wise
        for i in range(config.dd):
            for j in range(config.dl):
                ff = plt.figure()
                ax = plt.subplot(111)
                ax.plot(x_range, fs[:, i, j], label='f_'+str(i)+','+str(j), alpha=0.6)
                plt.xlabel('Epoch')
                plt.title('Mixing model weight ' + str(i) + ', ' + str(j) )
                pp.savefig(ff)
                plt.close()
         
    
    # prior means weights per epoch    
    if config.track_prior:
        if config.dl > 1 and config.dd > 1:
            fs = []
            for f in f_i:
                m = torch.load(f, map_location=device)['model_state_dict']
                fs.append(m['prior_mean.fc.0.weight'].cpu().detach().numpy())
            fs = np.asarray(fs)
            # labels ordered column-wise
            # create one plot for each component because there are so many
            for i in range(config.dl):
                for j in range(config.ns): 
                    ff = plt.figure()
                    ax = plt.subplot(111)       
                    ax.plot(x_range, fs[:, i, j], alpha=0.6)

                    plt.xlabel('Epoch')
                    plt.title('Prior mean weight, Component ' + str(i) + ', ' + str(j))
                    pp.savefig(ff)
                    plt.close()

    pp.close()
    
    
def inference_plots(m_ckpt, config, check_freq=100, pdf_name='inference.pdf'):
    '''
    Plot inference model weights: means (g) and log-variances (logv).
    m_ckpt: (str) Directory where all the checkpoints are.
    config: (namespace) config file, already loaded.
    check_freq: (int) frequency in which the checkpoint files were saved, e.g. every 100 epochs.
    pdf_name: (str) name to save the pdf file containing the plots.
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    pp = PdfPages(pdf_name)

    f_i = glob.glob(m_ckpt)
#     f_i.sort(key=os.path.getmtime)
    f_i.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    max_epoch = int(f_i[-1].split('/')[-1].split('.')[0].split('_')[-1])
    x_range = np.arange(1, max_epoch+1)
    x_range = x_range[0::check_freq]

    ## means
    fs = []
    # for i in range(max_epoch):
    for f in f_i:
        m = torch.load(f, map_location=device)['model_state_dict']
        fs.append(m['g.fc.0.weight'].cpu().detach().numpy())
    fs = np.asarray(fs)

    if config.dd > 1 and config.dl > 1:
        for i in range(config.dl):
            for j in range(config.ns+config.dd):
                ff = plt.figure()
                ax = plt.subplot(111)
                ax.plot(x_range, fs[:, i, j], label='f_'+str(i)+','+str(j), alpha=0.7)
                plt.xlabel('Epoch')
                plt.title('Inference mean model (g) weight ' + str(i) + ', ' + str(j) )
                pp.savefig(ff)
    #             plt.show()
                plt.close()

    ## log-var
    fs = []
    # for i in range(max_epoch):
    for f in f_i:
        m = torch.load(f, map_location=device)['model_state_dict']
        fs.append(m['logv.fc.0.weight'].cpu().detach().numpy())
    fs = np.asarray(fs)

    if config.dl > 1 and config.dd > 1:
        # labels ordered column-wise
        for i in range(config.dl):
            for j in range(config.ns+config.dd):
                ff = plt.figure()
                ax = plt.subplot(111)
                ax.plot(x_range, fs[:, i, j], label='f_'+str(i)+','+str(j), alpha=0.7)
                plt.xlabel('Epoch')
                plt.title('Inference log-variance model (logv) weight ' + str(i) + ', ' + str(j) )
                pp.savefig(ff)
    #             plt.show()
                plt.close()

    pp.close()
    
    
def gradient_plots(config, dset, m_ckpt, model, pdf_name='grad.pdf', check_freq=100):
    '''
    Plot the gradient wrt each weight.
    m_ckpt: (str) Directory where all the checkpoints are.
    config: (namespace) config file, already loaded.
    model: (discrete_simple_IVAE) current model. Contains e.g. whether the parameters were fixed.
    check_freq: (int) frequency in which the checkpoint files were saved, e.g. every 100 epochs.
    pdf_name: (str) name to save the pdf file containing the plots.
    dset: (SyntheticDataset(Dataset)) test set
    '''
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    f_i = glob.glob(m_ckpt)
#     f_i.sort(key=os.path.getmtime)
    f_i.sort(key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
    d_data, d_latent, d_aux = dset.get_dims()

    names = []
    shapes = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
            shapes.append(param.shape)

    titles = []
    for a in range(len(shapes)):
        for i in range(shapes[a][0]):
            try:
                for j in range(shapes[a][1]):
                    title = names[a] + '_' + str(i) + '_' + str(j)
                    titles.append(title)
            except:
                title = names[a] + '_' + str(i)
                titles.append(title)

    max_epoch = int(f_i[-1].split('/')[-1].split('.')[0].split('_')[-1])
    x_range = np.arange(1, max_epoch+1)
    x_range = x_range[0::check_freq]

    fs = []
    for f in f_i:
        g = torch.load(f, map_location=device)['grad']
        grads_together = torch.nn.utils.parameters_to_vector(g)
        grads_together = grads_together.detach().cpu().numpy()
        fs.append(grads_together)
    fs = np.asarray(fs)

    # plotting
    pp = PdfPages(pdf_name)
    for j in range(len(fs[0])):
        f = plt.figure()
        plt.plot(x_range, fs[:, j])
        plt.title(titles[j])
        pp.savefig(f)
        plt.close()
        
    pp.close()