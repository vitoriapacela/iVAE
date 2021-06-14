from numbers import Number

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

torch.set_default_dtype(torch.double)

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    
def weights_init(m):
    if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight)
        nn.init.xavier_uniform_(m.weight.data)
#         nn.init.kaiming_uniform_(m.weight.data)

def transf_logistic(phi):
    return (torch.exp(-phi)) / ((1 + torch.exp(-phi))**2)

def _check_inputs(size, mu, v):
    """
    Helper function to ensure inputs are compatible.
    
    Used to convert a number into a tensor, such as for the decoder variance.
    """
    if size is None and mu is None and v is None:
        raise ValueError("inputs can't all be None")
    elif size is not None:
        if mu is None:
            mu = torch.Tensor([0])
        if v is None:
            v = torch.Tensor([1])
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        v = v.expand(size)
        mu = mu.expand(size)
        return mu, v
    elif mu is not None and v is not None:
        if isinstance(v, Number):
            v = torch.Tensor([v]).type_as(mu)
        if v.size() != mu.size():
            v = v.expand(mu.size())
        return mu, v
    elif mu is not None:
        v = torch.Tensor([1]).type_as(mu).expand(mu.size())
        return mu, v
    elif v is not None:
        mu = torch.Tensor([0]).type_as(v).expand(v.size())
        return mu, v
    else:
        raise ValueError('Given invalid inputs: size={}, mu_logsigma={})'.format(size, (mu, v)))


def log_normal(x, mu=None, v=None, broadcast_size=False):
    """
    Compute the log-pdf of a normal distribution with diagonal covariance.
    
    # Notice that this is the univariate case, which is applied to all the vectors.
    
    x: (torch.Tensor) observationsor or latent representation, depending on the use
    (e.g. shape torch.Size([batch_dimension, 2]))
    mu: (torch.Tensor) mean (e.g. shape torch.Size([batch_dimension, 2]))
    v: (torch.Tensor) variance (e.g. shape torch.Size([batch_dimension, 2]))
    return: (torch.Tensor) log-pdf (e.g. shape torch.Size([batch_dimension, 2]))
    """
    if not broadcast_size:
        mu, v = _check_inputs(None, mu, v)
    else:
        mu, v = _check_inputs(x.size(), mu, v)
    assert mu.shape == v.shape
    return -0.5 * (np.log(2 * np.pi) + v.log() + (x - mu).pow(2).div(v))


def log_laplace(x, mu, b, broadcast_size=False):
    """compute the log-pdf of a laplace distribution with diagonal covariance"""
    # b might not have batch_dimension. This case is handled by _check_inputs
    if broadcast_size:
        mu, b = _check_inputs(x.size(), mu, b)
    else:
        mu, b = _check_inputs(None, mu, b)
    return -torch.log(2 * b) - (x - mu).abs().div(b)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation='none', slope=.1, device='cpu', bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.device = device
        self.bias = bias
        
        if isinstance(hidden_dim, Number):
            self.hidden_dim = [hidden_dim] * (self.n_layers - 1)
        elif isinstance(hidden_dim, list):
            self.hidden_dim = hidden_dim
        else:
            raise ValueError('Wrong argument type for hidden_dim: {}'.format(hidden_dim))

        if isinstance(activation, str):
            self.activation = [activation] * (self.n_layers - 1)
        elif isinstance(activation, list):
            self.hidden_dim = activation
        else:
            raise ValueError('Wrong argument type for activation: {}'.format(activation))

        self._act_f = []
        for act in self.activation:
            if act == 'lrelu':
                self._act_f.append(lambda x: F.leaky_relu(x, negative_slope=slope))
            elif act == 'xtanh':
                self._act_f.append(lambda x: self.xtanh(x, alpha=slope))
            elif act == 'sigmoid':
                self._act_f.append(F.sigmoid)
            elif act == 'none':
                self._act_f.append(lambda x: x)
            else:
                ValueError('Incorrect activation: {}'.format(act))

        if self.n_layers == 1:
            _fc_list = [nn.Linear(self.input_dim, self.output_dim, self.bias)]
        else:
            _fc_list = [nn.Linear(self.input_dim, self.hidden_dim[0], self.bias)]
            for i in range(1, self.n_layers - 1):
                _fc_list.append(nn.Linear(self.hidden_dim[i - 1], self.hidden_dim[i], self.bias))
            _fc_list.append(nn.Linear(self.hidden_dim[self.n_layers - 2], self.output_dim, self.bias))
        self.fc = nn.ModuleList(_fc_list)
        self.to(self.device)
        
    @staticmethod
    def xtanh(x, alpha=.1):
        """tanh function plus an additional linear term"""
        return x.tanh() + alpha * x

    def forward(self, x):
        '''
        Use activation functions after computing the hidden layer transformations,
        but no activation function is use at the end (after the transformation of the last layer).
        '''
        h = x
        for c in range(self.n_layers):
            if c == self.n_layers - 1:
                h = self.fc[c](h)
            else:
                h = self._act_f[c](self.fc[c](h))
        return h
        
class discrete_simple_IVAE(nn.Module):
    """
    Assumes uncentered prior means!
    This models simplifies the iVAE such that it can be interpreted as "conditional ICA".
    
    - Inference model (encoder, estimated posterior):
        - g:
              inputs: x, u (observations)
              output: mean of each component -- to be used to estimate the sources in the reparametrization trick
              default: neural network
              alternative: linear transformation, no activation function.
        - logv:
              inputs: x, u (observations)
              output: log-variance of each component -- to be used to estimate the sources in the reparametrization trick
              default: neural network
              alternative: linear transformation, no activation function.
              
    - Prior model:
        - prior_mean:
            input: u (additionally observed variable) -- usually a categorical variable one-hot encoded
            output: mean for each component
            default: linear transformation
            alternative: neural network
        - logl:
            input: u (additionally observed variable)
            output: log-variance for each component
            default: linear transformation
            alternative: neural network
            
    - Mixing model ("decoder"):
        - f:
            input: latent variables / estimated sources after the reparameterization trick (of each component)
            output: P(x=1) probability that the observation is 1
            default: linear transformation, no bias parameters.
    
    """
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1, f_bias=False, anneal_params=False, interpret=False, simple_g=False, device='cpu', simple_prior=True, simple_logv=False, fix_v=False, logvar=0.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.f_bias = f_bias
        self.anneal_params = anneal_params
        self.save_terms = interpret
        self.simple_g = simple_g
        self.device = device
        self.simple_prior = simple_prior
        self.decoder_dist = Bernoulli(device=device)
        self.simple_logv = simple_logv
        self.fix_v = fix_v
        self.logvar = logvar
        
        ## prior params
        if self.simple_prior:
            self.prior_mean = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=data_dim, n_layers=1, activation='none', slope=slope)    
            self.logl = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=data_dim, n_layers=1, activation='none', slope=slope)
        else:
            self.prior_mean = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)    
            self.logl = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)
        
        ## decoder params
        self.f = MLP(input_dim=latent_dim, output_dim=data_dim, hidden_dim=data_dim, n_layers=1, activation='none', bias=self.f_bias, slope=slope)
        
        ## encoder params
        # means
        if self.simple_g:
            # simple MLP
            self.g = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=latent_dim, n_layers=1, activation='none', bias=True, slope=slope)
#             self.g = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=latent_dim, n_layers=1, activation='none', bias=self.f_bias, slope=slope)
#             self.g = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=latent_dim, n_layers=1, activation=activation, bias=self.f_bias, slope=slope)
        else:
            # high-capacity neural networks
            self.g = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)
        
        # variances
        if not fix_v:
            if self.simple_logv:
                self.logv = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=latent_dim, n_layers=1, activation='none', slope=slope)
            else:
                self.logv = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)
        else:
            self.logv = torch.tensor(self.logvar, device=device)
        
    @staticmethod
    def reparameterize(mu, v):
        """
        Reparameterization trick.
        epsilon \sim N(0,1)
        z = mu + std*epsilon, where * and + are element-wise.
        
        mu: (torch.Tensor) mean (e.g. shape torch.Size([batch_dimension, n_components]))
        v: (torch.Tensor) variance (e.g. shape torch.Size([batch_dimension, n_components]))
        return: (torch.Tensor) z (e.g. shape torch.Size([batch_dimension, n_components]))
        """
        eps = torch.randn_like(mu)
#         scaled = eps.mul(v.sqrt())
#         return scaled.add(mu)   
        std = torch.exp(0.5 * torch.log(v))
        return mu + eps * std

    def encoder(self, x, u):
        """
        Inference model / estimated posterior.
        g maps the observations to the latent representation (mean).
        logv maps the observations to a latent representation (log-variance).
        
        x: (torch.Tensor) obsevations (e.g. shape torch.Size([batch_dimension, n_components]))
        u: (torch.Tensor) auxiliary observations (e.g. shape torch.Size([batch_dimension, n_components]))
        return: (list of torch.Tensor) g and v (variance) 
        (e.g. each element of the list with shape torch.Size([batch_dimension, 2]))
        """
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        if not self.fix_v:
            logv = self.logv(xu)
        else: 
            logv = self.logv
        return g, logv.exp()

    def decoder(self, s):
        """
        Mixing model.
        Maps the latent representation (estimated sources) to the observations probabilities.
        
        s: (torch.Tensor) estimated sources after reparameterization trick (e.g. shape torch.Size([batch_dimension, n_components]))
        return: (torch.Tensor) observation probability p(x=1) (e.g. shape torch.Size([batch_dimension, n_components]))
        """
        f = self.f(s)
        return torch.sigmoid(f)

    def prior(self, u):
        """
        m: Maps the auxiliary variable to the latent representation of the means.
        logl: Maps the auxiliary variable to the latent representation of the log-variances.
        
        u: (torch.Tensor) auxiliary observations (e.g. shape torch.Size([batch_dimension, n_components]))
        return: (torch.Tensor) mean and logl -- latents (e.g. shape torch.Size([batch_dimension, n_components]))
        """
        logl = self.logl(u)
        m = self.prior_mean(u)
        return m, logl.exp()

    def forward(self, x, u):
        m, l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s, m, l

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        '''
        return: (float) ELBO loss and (torch.Tensor) latent representation of mean z
        '''
        f, g, v, z, m_p, l = self.forward(x, u)
        
        M, d_latent = z.size()
         
        # p(x|z,u) decoder / mixing model
        logpx = self.decoder_dist.log_pmf(x, f, reduce=False).sum(dim=-1)
        
        # p(z|x,u) encoder / inference model
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        
        # p(z|u) prior
        logps_cu = log_normal(z, m_p, l).sum(dim=-1)
        
        
        elbo = -(logpx + logps_cu - logqs_cux).mean()
        
        # do not use inference and prior loss for debugging:
#         elbo = -(logpx).mean()

        if self.save_terms:
            return elbo, z, -logpx.mean(), -logps_cu.mean(), logqs_cux.mean()
        else:
            return elbo, z
        
        
#         BCE = -F.binary_cross_entropy(f, x, reduction='sum')
        
#         KLD = -0.5 * torch.sum(torch.log(l) - torch.log(v) - 1 + (g - self.prior_mean).pow(2) / l + v / l)
        
#         elbo = (BCE + KLD) / x.size(0)  # average per batch
#         return elbo, z


class u_simple_IVAE(nn.Module):
    """
    Assumes uncentered prior means!
    This models simplifies the iVAE such that it can be interpreted as "conditional ICA". That is, a couple of its neural networks are replaced by linear transformations.
    
    The encoder is composed of two models: g and logv.
    The encoder modeling the mean of the distribution (g) is a linear transformation (with no bias term and no activation function) implemented in the form of a MLP that receives the input data (observations) and outputs a the desired latent representation (estimated sources). The parameters of this MLP should be equivalent to the unmixing matrix.
    The encoder modeling the variance of the distribution (logv) has high capacity, it is a neural network with at least one hidden layer that receives the observations and outputs a vector of the desired latent dimension.
    The decoder (f) is a MLP with the reverse architecture of g. It maps the latent representation (estimated sources) to the reconstructed observations. The parameters of this MLP should be equivalent to a mixing matrix. The variance in this case is fixed to 0.01.
    In addition, there is a neural network modeling the prior variance (logl), which maps the auxiliary variable to the latent representation. The prior mean is fixed to 0.
    
    """
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1, f_bias=False, anneal_params=False, interpret=False, simple_g=False, simple_prior=True, simple_logv=False, noise_level=0.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.f_bias = f_bias
        self.anneal_params = anneal_params
        self.save_terms = interpret
        self.simple_g = simple_g
        self.simple_prior = simple_prior
        self.simple_logv = simple_logv
        self.noise_level = noise_level # (noise std)

        ## prior params
        if self.simple_prior:
            self.prior_mean = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=data_dim, n_layers=1, activation='none', slope=slope)    
            self.logl = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=data_dim, n_layers=1, activation='none', slope=slope)
        else:
            self.prior_mean = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)    
            self.logl = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)
        
        ## decoder params
        self.f = MLP(input_dim=latent_dim, output_dim=data_dim, hidden_dim=data_dim, n_layers=1, activation='none', bias=self.f_bias, slope=slope)
        self.decoder_var = (self.noise_level**2) * torch.ones(1)
        
        ## encoder params
        if self.simple_g:
            # simple MLP
            self.g = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=latent_dim, n_layers=1, activation='none', bias=self.f_bias, slope=slope)
        else:
        # high-capacity neural networks
            self.g = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)
            
        if self.simple_logv:
            self.logv = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=latent_dim, n_layers=1, activation='none', slope=slope)
        else:
            self.logv = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)
        
        
    @staticmethod
    def reparameterize(mu, v):
        """
        Reparameterization trick.
        epsilon \sim N(0,1)
        z = mu + std*epsilon, where * and + are element-wise.
        
        mu: (torch.Tensor) mean (e.g. shape torch.Size([batch_dimension, 2]))
        v: (torch.Tensor) variance (e.g. shape torch.Size([batch_dimension, 2]))
        return: (torch.Tensor) z (e.g. shape torch.Size([batch_dimension, 2]))
        """
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        """
        g maps the observations (mean) to the latent representation (estimated sources).
        logv maps the observations variance to a latent representation.
        
        x: (torch.Tensor) obsevations (e.g. shape torch.Size([batch_dimension, 2]))
        u: (torch.Tensor) auxiliary observations (e.g. shape torch.Size([batch_dimension, 2]))
        return: (list of torch.Tensor) g and v (variance) 
        (e.g. each element of the list with shape torch.Size([batch_dimension, 2]))
        """
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder(self, s):
        """
        Maps the latent representation (estimated sources) to the reconstructed observations.
        
        s: (torch.Tensor) latent vector (e.g. shape torch.Size([batch_dimension, 2]))
        return: (torch.Tensor) reconstruction (e.g. shape torch.Size([batch_dimension, 2]))
        """
        f = self.f(s)
        return f

    def prior(self, u):
        """
        Models the prior variance and maps the auxiliary variable to the latent representation.
        
        u: (torch.Tensor) auxiliary observations (e.g. shape torch.Size([batch_dimension, 2]))
        return: (torch.Tensor) latent representation of prior (e.g. shape torch.Size([batch_dimension, 2]))
        """
        m = self.prior_mean(u)
        logl = self.logl(u)
        return m, logl.exp()

    def forward(self, x, u):
        m, l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s, m, l

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        '''
        return: (float) ELBO loss and (torch.Tensor) latent representation of mean z
        '''
        f, g, v, z, m_p, l = self.forward(x, u)
        M, d_latent = z.size()
        
        # p(x|z,u) decoder / mixing model
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        # p(z|x,u) encoder / inference model
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        # p(z|u) prior
        logps_cu = log_normal(z, m_p, l).sum(dim=-1)
           
            
        var_diff = torch.var(z, axis=0) - torch.ones(d_latent).to(z.device)
        mean_diff = torch.mean(z, axis=0)
        # mean across all the data points in the batch:
        elbo = -(logpx + logps_cu - logqs_cux).mean()

        if self.save_terms:
            return elbo, z, -logpx.mean(), -logps_cu.mean(), logqs_cux.mean()
        else:
            return elbo, z

    
class simple_IVAE(nn.Module):
    """
    Assumes centered prior means!
    This models simplifies the iVAE such that it can be interpreted as "conditional ICA". That is, a couple of its neural networks are replaced by linear transformations.
    
    The encoder is composed of two models: g and logv.
    The encoder modeling the mean of the distribution (g) is a linear transformation (with no bias term and no activation function) implemented in the form of a MLP that receives the input data (observations) and outputs a the desired latent representation (estimated sources). The parameters of this MLP should be equivalent to the unmixing matrix.
    The encoder modeling the variance of the distribution (logv) has high capacity, it is a neural network with at least one hidden layer that receives the observations and outputs a vector of the desired latent dimension.
    The decoder (f) is a MLP with the reverse architecture of g. It maps the latent representation (estimated sources) to the reconstructed observations. The parameters of this MLP should be equivalent to a mixing matrix. The variance in this case is fixed to 0.01.
    In addition, there is a neural network modeling the prior variance (logl), which maps the auxiliary variable to the latent representation. The prior mean is fixed to 0.
    
    """
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1, f_bias=False, anneal_params=False, interpret=False, simple_g=False, simple_prior=True, simple_logv=False, noise_level=0.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.f_bias = f_bias
        self.anneal_params = anneal_params
        self.save_terms = interpret
        self.simple_g = simple_g
        self.simple_prior = simple_prior
        self.simple_logv = simple_logv
        self.noise_level = noise_level

        # prior params
        self.prior_mean = torch.zeros(1)
        
        if self.simple_prior:   
            self.logl = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=data_dim, n_layers=1, activation='none', slope=slope)
        else:  
            self.logl = MLP(input_dim=aux_dim, output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)
        
        # decoder params
        self.f = MLP(input_dim=latent_dim, output_dim=data_dim, hidden_dim=data_dim, n_layers=1, activation='none', bias=self.f_bias, slope=slope)
        self.decoder_var = (self.noise_level**2) * torch.ones(1)
        
        # encoder params
        if self.simple_g:
            # simple MLP
            self.g = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=latent_dim, n_layers=1, activation='none', bias=self.f_bias, slope=slope)
        else:
        # high-capacity neural network
            self.g = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)
          
        if self.simple_logv:
            self.logv = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=latent_dim, n_layers=1, activation='none', slope=slope)
        else:
            self.logv = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=hidden_dim, n_layers=n_layers, activation=activation, slope=slope)
        
        
    @staticmethod
    def reparameterize(mu, v):
        """
        Reparameterization trick.
        epsilon \sim N(0,1)
        z = mu + std*epsilon, where * and + are element-wise.
        
        mu: (torch.Tensor) mean (e.g. shape torch.Size([batch_dimension, 2]))
        v: (torch.Tensor) variance (e.g. shape torch.Size([batch_dimension, 2]))
        return: (torch.Tensor) z (e.g. shape torch.Size([batch_dimension, 2]))
        """
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        """
        g maps the observations (mean) to the latent representation (estimated sources).
        logv maps the observations variance to a latent representation.
        
        x: (torch.Tensor) obsevations (e.g. shape torch.Size([batch_dimension, 2]))
        u: (torch.Tensor) auxiliary observations (e.g. shape torch.Size([batch_dimension, 2]))
        return: (list of torch.Tensor) g and v (variance) 
        (e.g. each element of the list with shape torch.Size([batch_dimension, 2]))
        """
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder(self, s):
        """
        Maps the latent representation (estimated sources) to the reconstructed observations.
        
        s: (torch.Tensor) latent vector (e.g. shape torch.Size([batch_dimension, 2]))
        return: (torch.Tensor) reconstruction (e.g. shape torch.Size([batch_dimension, 2]))
        """
        f = self.f(s)
        return f

    def prior(self, u):
        """
        Models the prior variance and maps the auxiliary variable to the latent representation.
        
        u: (torch.Tensor) auxiliary observations (e.g. shape torch.Size([batch_dimension, 2]))
        return: (torch.Tensor) latent representation of prior (e.g. shape torch.Size([batch_dimension, 2]))
        """
        logl = self.logl(u)
        return logl.exp()

    def forward(self, x, u):
        l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s, l

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        '''
        return: (float) ELBO loss and (torch.Tensor) latent representation of mean z
        '''
        f, g, v, z, l = self.forward(x, u)
        M, d_latent = z.size()
        
        # p(x|z,u) decoder / mixing model
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        
        # p(z|x,u) encoder / inference model
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        
        # p(z|u) prior
        logps_cu = log_normal(z, None, l).sum(dim=-1)

        if self.anneal_params:
            ## a, b, c, d are to mirror the hyperparameters used in beta-VAE and beta-TC-VAE
            
            # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
            logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
            logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)).mean()
            return elbo, z
            
        else:
            var_diff = torch.var(z, axis=0) - torch.ones(d_latent).to(z.device)
            mean_diff = torch.mean(z, axis=0)
            # mean across all the data points in the batch:
            elbo = -(logpx + logps_cu - logqs_cux).mean()

            if self.save_terms:
                return elbo, z, -logpx.mean(), -logps_cu.mean(), logqs_cux.mean()
            else:
                return elbo, z
            
    
class simple_cleanIVAE(nn.Module):
    """
    This models simplifies the iVAE such that it can be interpreted as "conditional ICA". That is, a couple of its neural networks are replaced by linear transformations.
    
    The encoder is composed of two models: g and logv.
    The encoder modeling the mean of the distribution (g) is a linear transformation (with no bias term and no activation function) implemented in the form of a MLP that receives the input data (observations) and outputs a the desired latent representation (estimated sources). The parameters of this MLP should be equivalent to the unmixing matrix.
    The encoder modeling the variance of the distribution (logv) has high capacity, it is a neural network with at least one hidden layer that receives the observations and outputs a vector of the desired latent dimension.
    The decoder (f) is a MLP with the reverse architecture of g. It maps the latent representation (estimated sources) to the reconstructed observations. The parameters of this MLP should be equivalent to a mixing matrix. The variance in this case is fixed to 0.1.
    In addition, there is a neural network modeling the prior variance (logl), which maps the auxiliary variable to the latent representation. The prior mean is fixed to 0.
    
    """
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1, g_bias=False, anneal_params=True):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.g_bias = g_bias
        self.anneal_params = anneal_params

        # prior params
        self.prior_mean = torch.zeros(1)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        
        # encoder params
        self.g = MLP(input_dim=(data_dim + aux_dim), output_dim=latent_dim, hidden_dim=latent_dim, n_layers=1, activation='none', bias=self.g_bias, slope=slope)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        
        
    @staticmethod
    def reparameterize(mu, v):
        """
        Reparameterization trick.
        epsilon \sim N(0,1)
        z = mu + std*epsilon, where * and + are element-wise.
        
        mu: (torch.Tensor) mean (e.g. shape torch.Size([batch_dimension, 2]))
        v: (torch.Tensor) variance (e.g. shape torch.Size([batch_dimension, 2]))
        return: (torch.Tensor) z (e.g. shape torch.Size([batch_dimension, 2]))
        """
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        """
        g maps the observations (mean) to the latent representation (estimated sources).
        logv maps the observations variance to a latent representation.
        
        x: (torch.Tensor) obsevations (e.g. shape torch.Size([batch_dimension, 2]))
        u: (torch.Tensor) auxiliary observations (e.g. shape torch.Size([batch_dimension, 2]))
        return: (list of torch.Tensor) g and v (variance) 
        (e.g. each element of the list with shape torch.Size([batch_dimension, 2]))
        """
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder(self, s):
        """
        Maps the latent representation (estimated sources) to the reconstructed observations.
        
        s: (torch.Tensor) latent vector (e.g. shape torch.Size([batch_dimension, 2]))
        return: (torch.Tensor) reconstruction (e.g. shape torch.Size([batch_dimension, 2]))
        """
        f = self.f(s)
        return f

    def prior(self, u):
        """
        Models the prior variance and maps the auxiliary variable to the latent representation.
        
        u: (torch.Tensor) auxiliary observations (e.g. shape torch.Size([batch_dimension, 2]))
        return: (torch.Tensor) latent representation of prior (e.g. shape torch.Size([batch_dimension, 2]))
        """
        logl = self.logl(u)
        return logl.exp()

    def forward(self, x, u):
        l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s, l

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        '''
        return: (float) ELBO loss and (torch.Tensor) latent representation of mean z
        '''
        f, g, v, z, l = self.forward(x, u)
        M, d_latent = z.size()
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps_cu = log_normal(z, None, l).sum(dim=-1)

        if self.anneal_params:
            ## a, b, c, d are to mirror the hyperparameters used in beta-VAE and beta-TC-VAE
            
            # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
            logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
            logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)).mean()
            
        else:
            # mean across all the data points in the batch
            elbo = (logpx + logps_cu - logqs_cux).mean()
        return elbo, z
    
    
class cleanIVAE(nn.Module):
    def __init__(self, data_dim, latent_dim, aux_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior params
        self.prior_mean = torch.zeros(1)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder(self, s):
        f = self.f(s)
        return f

    def prior(self, u):
        logl = self.logl(u)
        return logl.exp()

    def forward(self, x, u):
        l = self.prior(u)
        g, v = self.encoder(x, u)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s, l

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z, l = self.forward(x, u)
        M, d_latent = z.size()
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps_cu = log_normal(z, None, l).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)
        

        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps_cu)).mean()
        return elbo, z


class cleanVAE(nn.Module):

    def __init__(self, data_dim, latent_dim, n_layers=3, activation='xtanh', hidden_dim=50, slope=.1):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior_params
        self.prior_mean = torch.zeros(1)
        self.prior_var = torch.ones(1)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.decoder_var = .1 * torch.ones(1)
        # encoder params
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope)

    @staticmethod
    def reparameterize(mu, v):
        eps = torch.randn_like(mu)
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def encoder(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv.exp()

    def decoder(self, s):
        f = self.f(s)
        return f

    def forward(self, x):
        g, v = self.encoder(x)
        s = self.reparameterize(g, v)
        f = self.decoder(s)
        return f, g, v, s

    def elbo(self, x, u, N, a=1., b=1., c=1., d=1.):
        f, g, v, z = self.forward(x)
        M, d_latent = z.size()
        logpx = log_normal(x, f, self.decoder_var.to(x.device)).sum(dim=-1)
        logqs_cux = log_normal(z, g, v).sum(dim=-1)
        logps = log_normal(z, None, None, broadcast_size=True).sum(dim=-1)

        # no view for v to account for case where it is a float. It works for general case because mu shape is (1, M, d)
        logqs_tmp = log_normal(z.view(M, 1, d_latent), g.view(1, M, d_latent), v.view(1, M, d_latent))
        logqs = torch.logsumexp(logqs_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
        logqs_i = (torch.logsumexp(logqs_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

        elbo = -(a * logpx - b * (logqs_cux - logqs) - c * (logqs - logqs_i) - d * (logqs_i - logps)).mean()
        return elbo, z


class Discriminator(nn.Module):
    def __init__(self, z_dim=5, hdim=1000):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hdim, 2),
        )
        self.hdim = hdim

    def forward(self, z):
        return self.net(z).squeeze()


def permute_dims(z):
    assert z.dim() == 2

    B, _ = z.size()
    perm_z = []
    for z_j in z.split(1, 1):
        perm = torch.randperm(B).to(z.device)
        perm_z_j = z_j[perm]
        perm_z.append(perm_z_j)

    return torch.cat(perm_z, 1)


class Dist:
    def __init__(self):
        pass

    def sample(self, *args):
        pass

    def log_pdf(self, *args, **kwargs):
        pass


class Normal(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.c = 2 * np.pi * torch.ones(1).to(self.device)
        self._dist = dist.normal.Normal(torch.zeros(1).to(self.device), torch.ones(1).to(self.device))
        self.name = 'gauss'

    def sample(self, mu, v):
        eps = self._dist.sample(mu.size()).squeeze()
        scaled = eps.mul(v.sqrt())
        return scaled.add(mu)

    def log_pdf(self, x, mu, v, reduce=True, param_shape=None):
        """compute the log-pdf of a normal distribution with diagonal covariance"""
        if param_shape is not None:
            mu, v = mu.view(param_shape), v.view(param_shape)
        lpdf = -0.5 * (torch.log(self.c) + v.log() + (x - mu).pow(2).div(v))
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf

    def log_pdf_full(self, x, mu, v):
        """
        compute the log-pdf of a normal distribution with full covariance
        v is a batch of "pseudo sqrt" of covariance matrices of shape (batch_size, d_latent, d_latent)
        mu is batch of means of shape (batch_size, d_latent)
        """
        batch_size, d = mu.size()
        cov = torch.einsum('bik,bjk->bij', v, v)  # compute batch cov from its "pseudo sqrt"
        assert cov.size() == (batch_size, d, d)
        inv_cov = torch.inverse(cov)  # works on batches
        c = d * torch.log(self.c)
        # matrix log det doesn't work on batches!
        _, logabsdets = self._batch_slogdet(cov)
        xmu = x - mu
        return -0.5 * (c + logabsdets + torch.einsum('bi,bij,bj->b', [xmu, inv_cov, xmu]))

    def _batch_slogdet(self, cov_batch: torch.Tensor):
        """
        compute the log of the absolute value of determinants for a batch of 2D matrices. Uses torch.slogdet
        this implementation is just a for loop, but that is what's suggested in torch forums
        gpu compatible
        """
        batch_size = cov_batch.size(0)
        signs = torch.empty(batch_size, requires_grad=False).to(self.device)
        logabsdets = torch.empty(batch_size, requires_grad=False).to(self.device)
        for i, cov in enumerate(cov_batch):
            signs[i], logabsdets[i] = torch.slogdet(cov)
        return signs, logabsdets


class Laplace(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.laplace.Laplace(torch.zeros(1).to(self.device), torch.ones(1).to(self.device) / np.sqrt(2))
        self.name = 'laplace'

    def sample(self, mu, b):
        eps = self._dist.sample(mu.size())
        scaled = eps.mul(b)
        return scaled.add(mu)

    def log_pdf(self, x, mu, b, reduce=True, param_shape=None):
        """compute the log-pdf of a laplace distribution with diagonal covariance"""
        if param_shape is not None:
            mu, b = mu.view(param_shape), b.view(param_shape)
        lpdf = -torch.log(2 * b) - (x - mu).abs().div(b)
        if reduce:
            return lpdf.sum(dim=-1)
        else:
            return lpdf


class Bernoulli(Dist):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self._dist = dist.bernoulli.Bernoulli(0.5 * torch.ones(1).to(self.device))
        self.name = 'bernoulli'

    def sample(self, p):
        eps = self._dist.sample(p.size())
        return eps

    def log_pmf(self, x, f, reduce=True, param_shape=None):
        """compute the log-pmf of a bernoulli distribution"""
        if param_shape is not None:
            f = f.view(param_shape)
        lpmf = x * torch.log(f) + (1 - x) * torch.log(1 - f)
        if reduce:
            return lpmf.sum(dim=-1)
        else:
            return lpmf


class iVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu', anneal=False):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal

        if prior is None:
            self.prior_dist = Normal(device=device)
        else:
            self.prior_dist = prior

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def encoder_params(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv.exp()

    def decoder_params(self, s):
        f = self.f(s)
        return f, self.decoder_var

    def prior_params(self, u):
        logl = self.logl(u)
        return self.prior_mean, logl.exp()

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, prior_params

    def elbo(self, x, u):
        decoder_params, (g, v), z, prior_params = self.forward(x, u)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder_dist.log_pdf(z, g, v)
        log_pz_u = self.prior_dist.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder_dist.log_pdf(z.view(M, 1, self.latent_dim), g.view(1, M, self.latent_dim),
                                                   v.view(1, M, self.latent_dim), reduce=False)
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z

        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder_var.item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False


class DiscreteIVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim,
                 n_layers=2, hidden_dim=20, activation='lrelu', slope=.1, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # prior_params
        self.prior_mean = torch.zeros(1).to(device)
        self.logl = MLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # encoder params
        self.g = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

    def encoder_params(self, x, u):
        xu = torch.cat((x, u), 1)
        g = self.g(xu)
        logv = self.logv(xu)
        return g, logv

    def decoder_params(self, z):
        f = self.f(z)
        return torch.sigmoid(f)

    def prior_params(self, u):
        logl = self.logl(u)
        return self.prior_mean, logl

    def forward(self, x, u):
        prior_params = self.prior_params(u)
        encoder_params = self.encoder_params(x, u)
        z = self.reparameterize(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, prior_params

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def elbo(self, x, u):
        f, (g, logv), z, (h, logl) = self.forward(x, u)
        BCE = -F.binary_cross_entropy(f, x, reduction='sum')
        l = logl.exp()
        v = logv.exp()
        KLD = -0.5 * torch.sum(logl - logv - 1 + (g - h).pow(2) / l + v / l)
        return (BCE + KLD) / x.size(0), z  # average per batch


class VAE(nn.Module):

    def __init__(self, latent_dim, data_dim, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        if decoder is None:
            self.decoder_dist = Normal(device=device)
        else:
            self.decoder_dist = decoder

        if encoder is None:
            self.encoder_dist = Normal(device=device)
        else:
            self.encoder_dist = encoder
        self.prior_dist = Normal(device)
        self.prior_mean = torch.zeros(1).to(device)
        self.prior_var = torch.ones(1).to(device)

        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.decoder_var = .01 * torch.ones(1).to(device)
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)

    def encoder_params(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv.exp()

    def decoder_params(self, s):
        f = self.f(s)
        return f, self.decoder_var

    def forward(self, x):
        encoder_params = self.encoder_params(x)
        z = self.encoder_dist.sample(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z, (self.prior_mean, self.prior_var)

    def elbo(self, x):
        decoder_params, encoder_params, z, prior_params = self.forward(x)
        log_px_z = self.decoder_dist.log_pdf(x, *decoder_params)
        log_qz_x = self.encoder_dist.log_pdf(z, *encoder_params)
        log_pz = self.prior_dist.log_pdf(z, *prior_params)

        return (log_px_z + log_pz - log_qz_x).mean(), z


class DiscreteVAE(nn.Module):
    def __init__(self, latent_dim, data_dim,
                 n_layers=2, hidden_dim=20, activation='lrelu', slope=.1, device='cpu'):
        super().__init__()
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope

        # decoder params
        self.f = MLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope, device=device)
        # encoder params
        self.g = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                     device=device)
        self.logv = MLP(data_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                        device=device)

    def encoder_params(self, x):
        g = self.g(x)
        logv = self.logv(x)
        return g, logv

    def decoder_params(self, z):
        f = self.f(z)
        return torch.sigmoid(f)

    def forward(self, x):
        encoder_params = self.encoder_params(x)
        z = self.reparameterize(*encoder_params)
        decoder_params = self.decoder_params(z)
        return decoder_params, encoder_params, z

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

    def elbo(self, x):
        f, (g, logv), z = self.forward(x)
        BCE = -F.binary_cross_entropy(f, x, reduction='sum')
        v = logv.exp()
        KLD = 0.5 * torch.sum(logv + 1 - g.pow(2) - v)
        return (BCE + KLD) / x.size(0), z  # average per batch


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, slope, device, fixed_mean=None,
                 fixed_var=None):
        super().__init__()
        self.distribution = Normal(device=device)
        if fixed_mean is None:
            self.mean = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                            device=device)
        else:
            self.mean = lambda x: fixed_mean * torch.ones(1).to(device)
        if fixed_var is None:
            self.log_var = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                               device=device)
        else:
            self.log_var = lambda x: np.log(fixed_var) * torch.ones(1).to(device)

    def sample(self, *params):
        return self.distribution.sample(*params)

    def log_pdf(self, x, *params, **kwargs):
        return self.distribution.log_pdf(x, *params, **kwargs)

    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        return self.mean(x), self.log_var(x).exp()


class LaplaceMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, activation, slope, device, fixed_mean=None,
                 fixed_var=None):
        super().__init__()
        self.distribution = Laplace(device=device)
        if fixed_mean is None:
            self.mean = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                            device=device)
        else:
            self.mean = lambda x: fixed_mean * torch.ones(1).to(device)
        if fixed_var is None:
            self.log_var = MLP(input_dim, output_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                               device=device)
        else:
            self.log_var = lambda x: np.log(fixed_var) * torch.ones(1).to(device)

    def sample(self, *params):
        return self.distribution.sample(*params)

    def log_pdf(self, x, *params, **kwargs):
        return self.distribution.log_pdf(x, *params, **kwargs)

    def forward(self, *input):
        if len(input) > 1:
            x = torch.cat(input, dim=1)
        else:
            x = input[0]
        return self.mean(x), self.log_var(x).exp()


class ModularIVAE(nn.Module):
    def __init__(self, latent_dim, data_dim, aux_dim, prior=None, decoder=None, encoder=None,
                 n_layers=3, hidden_dim=50, activation='lrelu', slope=.1, device='cpu', anneal=False):
        super().__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.slope = slope
        self.anneal_params = anneal

        if prior is None:
            self.prior = GaussianMLP(aux_dim, latent_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                                     device=device, fixed_mean=0)
        else:
            self.prior = prior

        if decoder is None:
            self.decoder = GaussianMLP(latent_dim, data_dim, hidden_dim, n_layers, activation=activation, slope=slope,
                                       device=device, fixed_var=.01)
        else:
            self.decoder = decoder

        if encoder is None:
            self.encoder = GaussianMLP(data_dim + aux_dim, latent_dim, hidden_dim, n_layers, activation=activation,
                                       slope=slope, device=device)
        else:
            self.encoder = encoder

        self.apply(weights_init)

        self._training_hyperparams = [1., 1., 1., 1., 1]

    def forward(self, x, u):
        encoder_params = self.encoder(x, u)
        z = self.encoder.sample(*encoder_params)
        decoder_params = self.decoder(z)
        prior_params = self.prior(u)
        return decoder_params, encoder_params, prior_params, z

    def elbo(self, x, u):
        decoder_params, encoder_params, prior_params, z = self.forward(x, u)
        log_px_z = self.decoder.log_pdf(x, *decoder_params)
        log_qz_xu = self.encoder.log_pdf(z, *encoder_params)
        log_pz_u = self.prior.log_pdf(z, *prior_params)

        if self.anneal_params:
            a, b, c, d, N = self._training_hyperparams
            M = z.size(0)
            log_qz_tmp = self.encoder.log_pdf(z.view(M, 1, self.latent_dim), encoder_params, reduce=False,
                                              param_shape=(1, M, self.latent_dim))
            log_qz = torch.logsumexp(log_qz_tmp.sum(dim=-1), dim=1, keepdim=False) - np.log(M * N)
            log_qz_i = (torch.logsumexp(log_qz_tmp, dim=1, keepdim=False) - np.log(M * N)).sum(dim=-1)

            return (a * log_px_z - b * (log_qz_xu - log_qz) - c * (log_qz - log_qz_i) - d * (
                    log_qz_i - log_pz_u)).mean(), z
        else:
            return (log_px_z + log_pz_u - log_qz_xu).mean(), z

    def anneal(self, N, max_iter, it):
        thr = int(max_iter / 1.6)
        a = 0.5 / self.decoder.log_var(0).exp().item()
        self._training_hyperparams[-1] = N
        self._training_hyperparams[0] = min(2 * a, a + a * it / thr)
        self._training_hyperparams[1] = max(1, a * .3 * (1 - it / thr))
        self._training_hyperparams[2] = min(1, it / thr)
        self._training_hyperparams[3] = max(1, a * .5 * (1 - it / thr))
        if it > thr:
            self.anneal_params = False


# MNIST MODELS - OLD IMPLEMENTATION

class ConvolutionalVAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            # input is mnist image: 1x28x28
            nn.Conv2d(1, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.Conv2d(32, 128, 4, 2, 1),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.Conv2d(128, 512, 7, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )
        self.fc1 = nn.Linear(200, latent_dim)
        self.fc2 = nn.Linear(200, latent_dim)

        self.decoder = nn.Sequential(
            # input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.ConvTranspose2d(512, 128, 7, 1, 0),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1)  # 1x28x28
        )

    def encode(self, x):
        h = self.encoder(x.view(-1, 1, 28, 28)).squeeze()
        return self.fc1(F.relu(h)), self.fc2(F.relu(h))

    def decode(self, z):
        h = self.decoder(z.view(z.size(0), z.size(1), 1, 1))
        return torch.sigmoid(h.view(-1, 28 * 28))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparameterize(mu, logv)
        f = self.decode(z)
        return f, mu, logv


class VAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class ConvolutionalIVAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        # encoder
        self.encoder = nn.Sequential(
            # input is mnist image: 1x28x28
            nn.Conv2d(1, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.Conv2d(32, 128, 4, 2, 1),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.Conv2d(128, 512, 7, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.Conv2d(512, 200, 1, 1, 0),  # 200x1x1
        )
        self.fc1 = nn.Linear(200, latent_dim)
        self.fc2 = nn.Linear(200, latent_dim)

        # decoder
        self.decoder = nn.Sequential(
            # input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 1, 1, 0),  # 512x1x1
            nn.BatchNorm2d(512),  # 512x1x1
            nn.ReLU(inplace=True),  # 512x1x1
            nn.ConvTranspose2d(512, 128, 7, 1, 0),  # 128x7x7
            nn.BatchNorm2d(128),  # 128x7x7
            nn.ReLU(inplace=True),  # 128x7x7
            nn.ConvTranspose2d(128, 32, 4, 2, 1),  # 32x14x14
            nn.BatchNorm2d(32),  # 32x14x14
            nn.ReLU(inplace=True),  # 32x14x14
            nn.ConvTranspose2d(32, 1, 4, 2, 1)  # 1x28x28
        )

        # prior
        self.l1 = nn.Linear(10, 200)
        self.l21 = nn.Linear(200, latent_dim)
        self.l22 = nn.Linear(200, latent_dim)

    def encode(self, x):
        h = self.encoder(x.view(-1, 1, 28, 28)).squeeze()
        return self.fc1(F.relu(h)), self.fc2(F.relu(h))

    def decode(self, z):
        h = self.decoder(z.view(z.size(0), z.size(1), 1, 1))
        return torch.sigmoid(h.view(-1, 28 * 28))

    def prior(self, y):
        h2 = F.relu(self.l1(y))
        # h2 = self.l1(y)
        return self.l21(h2), self.l22(h2)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logv = self.encode(x)
        mup, logl = self.prior(y)
        z = self.reparameterize(mu, logv)
        f = self.decode(z)
        return f, mu, logv, mup, logl


class iVAEforMNIST(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()

        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, latent_dim)
        self.fc22 = nn.Linear(400, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

        hidden_dim = 200
        self.l1 = nn.Linear(10, hidden_dim)
        self.l21 = nn.Linear(hidden_dim, latent_dim)
        self.l22 = nn.Linear(hidden_dim, latent_dim)

    def encode(self, x):
        h1 = F.leaky_relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def prior(self, y):
        h2 = F.relu(self.l1(y))
        # h2 = self.l1(y)
        return self.l21(h2), self.l22(h2)

    def decode(self, z):
        h3 = F.leaky_relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mu, logvar = self.encode(x.view(-1, 784))
        mup, logl = self.prior(y)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, mup, logl