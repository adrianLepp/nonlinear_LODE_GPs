import gpytorch
import torch
from typing import List
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import GaussianLikelihood
from linear_operator.operators import DiagLinearOperator, ConstantDiagLinearOperator
import math

from nonlinear_LODE_GPs.lodegp import optimize_gp
from nonlinear_LODE_GPs.kernels import FeedbackControl_Kernel
from nonlinear_LODE_GPs.mean_modules import FeedbackControl_Mean

def exponential_Gaussian(gaussian: gpytorch.distributions.MultivariateNormal):
    # return gaussian
    mu = gaussian.mean
    Sigma = gaussian.covariance_matrix
    
    
    # exp_mu = torch.exp(mu + 0.5 * torch.diag(Sigma))  # shape: (d,)
    # outer_exp_mu = exp_mu.unsqueeze(1) * exp_mu.unsqueeze(0)  # shape: (d, d)
    # cov_Y = (torch.exp(Sigma) - 1) * outer_exp_mu  # shape: (d, d)
    # return gpytorch.distributions.MultivariateNormal(exp_mu, cov_Y)

    return gpytorch.distributions.MultivariateNormal(torch.exp(mu), gaussian._covar)

def squared_gaussian(gaussian: gpytorch.distributions.MultivariateNormal):
    # return gaussian
    alpha = 1
    mu = gaussian.mean
    Sigma = gaussian._covar # covariance_matrix # _covar

    mean = alpha + 1/2 * mu**2
    #covar = mu * Sigma.to_dense()  * mu
    covar = torch.diag(mu) @ Sigma @  torch.diag(mu)
    return gpytorch.distributions.MultivariateNormal(mean, covar) # + torch.eye(covar.shape[0]) * 1e-4
class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                  covar_module=None,
                  mean_module=None
                  ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module if mean_module is not None else gpytorch.means.ConstantMean()
        self.covar_module = covar_module if covar_module is not None else gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x:torch.Tensor, train_y:torch.Tensor, likelihood:gpytorch.likelihoods.Likelihood, num_tasks:int):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def estimate(self, x:torch.Tensor):
        return self.likelihood(self(x)).mean
    
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
    
    def estimate(self, x:torch.Tensor):
        return self.likelihood(self(x)).mean


def y_ref_from_alpha_beta(alpha:gpytorch.distributions.Distribution, beta:gpytorch.distributions.Distribution, u:torch.Tensor):
        mean_x =  alpha.mean + beta.mean * u
        covar_x = u.unsqueeze(0) * beta.covariance_matrix * u.unsqueeze(1) + alpha.covariance_matrix
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x + torch.eye(covar_x.shape[0]) * 1e-8)


# class _Linearizing_Control():
#     def __init__(self, train_x:torch.Tensor, train_u:torch.Tensor, train_y:torch.Tensor, likelihood, **kwargs):
#         pass

class Linearizing_Control(gpytorch.models.ExactGP):
    def __init__(self, train_x:torch.Tensor, train_y:torch.Tensor, likelihood):
        super().__init__(train_x,  train_y, likelihood)

        self.alpha = GP(train_x[:,0:-1], train_y, likelihood, mean_module=gpytorch.means.ZeroMean())
        self.beta = GP(train_x[:,0:-1], train_y, likelihood, covar_module=gpytorch.kernels.ConstantKernel())

    def forward(self, x:torch.Tensor):
        u = x[:,-1] .unsqueeze(0)
        mean_x =  self.beta.mean_module(x[:,0:-1])* u.squeeze() + self.alpha.mean_module(x[:,0:-1])
        covar_x = u * self.beta.covar_module(x[:,0:-1]) *u.transpose(0,1) + self.alpha.covar_module(x[:,0:-1])
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Linearizing_Control_2(torch.nn.Module):
    def __init__(
            self, 
            train_x:List[torch.Tensor],
            train_u:List[torch.Tensor],
            train_y:List[torch.Tensor], # y_ref
            consecutive_training = True,
            **kwargs
        ):
        super().__init__()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.cat(var, dim=0))

        self.likelihood = likelihood

        # self.likelihood = MultiOutputLikelihood(likelihood)

        self.consecutive = consecutive_training

        if consecutive_training:
            len_per_model = math.floor(len(train_x)/2)
            # if len(train_x) != 2:
            #     raise Warning('Consecutive training takes two training sets')


            self.alpha = GP(
                torch.cat(train_x[0:len_per_model], dim=0), 
                torch.cat(train_y[0:len_per_model], dim=0), 
                gpytorch.likelihoods.GaussianLikelihood(), 
                #mean_module=gpytorch.means.ZeroMean()
            )
            self._beta = GP(
                torch.cat(train_x[len_per_model::], dim=0), 
                torch.cat(train_y[len_per_model::], dim=0), 
                gpytorch.likelihoods.GaussianLikelihood(), 
                #mean_module=gpytorch.means.ZeroMean()
            )

            #self._beta = GP(train_x[1], train_y[1], gpytorch.likelihoods.GaussianLikelihood(), covar_module=gpytorch.kernels.ConstantKernel())    
            self.train_u = torch.cat(train_u[len_per_model::])
        else:
            self.train_x = torch.cat(train_x, dim=0)
            self.train_u = torch.cat(train_u, dim=0)
            self.train_y = torch.cat(train_y, dim=0)

            self.train_targets = torch.cat(train_y, dim=0)
            self.train_inputs = [torch.cat(train_x, dim=0)]
            
            self._alpha = GP(self.train_x, self.train_y, 
                             #gpytorch.likelihoods.GaussianLikelihood(), 
                             self.likelihood,
                             gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)))#, mean_module=gpytorch.means.ZeroMean())
            self._beta = GP(self.train_x, self.train_y, 
                            # gpytorch.likelihoods.GaussianLikelihood(), 
                            self.likelihood,
                            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)))#, covar_module=gpytorch.kernels.ConstantKernel())

    def optimize(self, training_iterations=100, verbose=False):
        if self.consecutive:
            self.optimize_consecutive(training_iterations, verbose)
        else:
            self.optimize_all(training_iterations, verbose)

    def optimize_all(self, training_iterations=100, verbose=False):
        self._alpha.train()
        self._alpha.likelihood.train()
        self._beta.train()
        self._beta.likelihood.train()

        optimizer = torch.optim.Adam(list(self._alpha.parameters()) + list(self._beta.parameters()), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self._alpha) # TODO: Does alpha or beta matter here?

        training_loss = [] 
        for i in range(training_iterations): 
            optimizer.zero_grad()

            output_alpha = self._alpha(self.train_x)
            log_beta = self._beta(self.train_x)
            output_beta = exponential_Gaussian(log_beta)
            # output_beta = squared_gaussian(log_beta)


            # output = self.likelihood(output_alpha, output_beta, self.train_u)
            # output = self.likelihood(output_alpha, log_beta, self.train_u)
            
            # output = self.likelihood(y_ref_from_alpha_beta(output_alpha, output_beta, self.train_u))#output_beta
            output = y_ref_from_alpha_beta(output_alpha, output_beta, self.train_u)

            loss = -mll(output, self.train_y )
            loss.backward()

            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, (loss.item())))
            
            optimizer.step()

            training_loss.append(loss.item())

        self._alpha.eval()
        self._alpha.likelihood.eval()
        self._beta.eval()
        self._beta.likelihood.eval()

        print("\n----------------------------------------------------------------------------------\n")
        print('Trained model parameters:')
        named_parameters = list(self.named_parameters())
        param_conversion = torch.nn.Softplus()

        for j in range(len(named_parameters)):
            print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()
            # print(named_parameters[j][0], (named_parameters[j][1].data)) #.item()
        print("\n----------------------------------------------------------------------------------\n")
        return training_loss 


    def optimize_consecutive(self, training_iterations=100, verbose=False):
        self.optimize_alpha(training_iterations, verbose)
        self._alpha.eval()
        self._alpha.likelihood.eval()

        train_targets_beta = (self._beta.train_targets - self._alpha(self._beta.train_inputs[0]).mean.detach().clone())/ self.train_u
        self._beta.set_train_data(self._beta.train_inputs[0], train_targets_beta.detach())

        self.optimize_beta(training_iterations, verbose)

        self._beta.eval()
        self._beta.likelihood.eval()

    def optimize_alpha(self,training_iterations:int=100, verbose=False):
        self._alpha.train()
        self._alpha.likelihood.train()
        optimizer = torch.optim.Adam(self._alpha.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._alpha.likelihood, self._alpha)
        training_loss = [] 
        for i in range(training_iterations): 
            optimizer.zero_grad()
            output = self._alpha(self._alpha.train_inputs[0])
            loss = -mll(output, self._alpha.train_targets)
            loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
            training_loss.append(loss.item())

    def optimize_beta(self,training_iterations:int=100, verbose=False):
        self._beta.train()
        self._beta.likelihood.train()
        optimizer = torch.optim.Adam(self._beta.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self._beta.likelihood, self._beta)
        training_loss = [] 
        for i in range(training_iterations): 
            optimizer.zero_grad()
            output = self._beta(self._beta.train_inputs[0]) #self._beta(self._beta.train_inputs[0])
            #loss = -mll(output, (self._beta.train_targets - self._alpha(self._beta.train_inputs[0]).mean)/ self.train_u)
            loss = -mll(output, self._beta.train_targets) 
            loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
            training_loss.append(loss.item())
    

    def y_ref(self, x:torch.Tensor, u:torch.Tensor):
        output = self(x, u)
        return output.mean
    
    def forward(self, x:torch.Tensor, u:torch.Tensor):
        alpha = self._alpha(x)
        log_beta = self._beta(x)
        beta = exponential_Gaussian(log_beta)
        # beta = squared_gaussian(log_beta)
        return y_ref_from_alpha_beta(alpha, beta, u)
    
    def get_nonlinear_fcts(self):
        self._beta.eval()
        self._alpha.eval()
    
        def alpha(x):
            return self._alpha(torch.tensor(x).unsqueeze(0)).mean.clone().detach().numpy()
            # with torch.no_grad:
            #       return control_gp.alpha(torch.tensor(x).unsqueeze(0)).mean.numpy()

        def beta(x,u):
            # log_beta = self._beta(torch.tensor(x).unsqueeze(0)).mean.clone().detach().numpy()
            log_beta = self._beta(torch.tensor(x).unsqueeze(0))
            beta = exponential_Gaussian(log_beta).mean.clone().detach().numpy()
            # beta = squared_gaussian(log_beta).mean.clone().detach().numpy()

            return beta
            # with torch.no_grad:
            # return control_gp.beta(torch.tensor(x).unsqueeze(0)).mean.numpy()

        return alpha, beta


class Linearizing_Control_3(torch.nn.Module):
    def __init__(
            self, 
            train_x:List[torch.Tensor],
            train_u:List[torch.Tensor],
            train_y:List[torch.Tensor],
            *args, **kwargs
        ):
        super().__init__()
        self.train_x = torch.cat(train_x, dim=0)
        self.train_u = torch.cat(train_u, dim=0)
        self.train_y = torch.cat(train_y, dim=0)

        self.alpha = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood(), mean_module=gpytorch.means.ZeroMean())
        self.beta = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood(), covar_module=gpytorch.kernels.ConstantKernel())
        self.model = gpytorch.models.IndependentModelList(self.alpha, self.beta)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(self.alpha.likelihood, self.beta.likelihood)

    def optimize(self, training_iterations=100, verbose=False):
        self.model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.3)

        training_loss = []

        for i in range(training_iterations):
        
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)
            y_ref = y_ref_from_alpha_beta(*output, self.train_u)
            #output.mean[0] + output.mean[1] * self.model.train_inputs[:,-1] 
            loss = -mll([y_ref, y_ref], self.model.train_targets)
            loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

            training_loss.append(loss.item())

    def forward(self, x:torch.Tensor, u:torch.Tensor):
        output = self.model(x, x)
        y_ref:gpytorch.distributions.MultivariateNormal = y_ref_from_alpha_beta(*output, u)
        return y_ref

    def y_ref(self, x:torch.Tensor, u:torch.Tensor):
        return self(x,u).mean



class Linearizing_Control_4(gpytorch.models.ExactGP):
    def __init__(self, x:torch.Tensor, u:torch.Tensor, y_ref:torch.Tensor,  likelihood, b:float, controller, variance, **kwargs):
        train_x = torch.cat(x, dim=0)
        train_u = torch.cat(u, dim=0)
        train_y = torch.cat(y_ref, dim=0)
        super().__init__(train_x,  train_y, likelihood)
        
        self.train_u = train_u
        self.U = DiagLinearOperator(self.train_u)
        self.Var = DiagLinearOperator(torch.cat(variance, dim=0))

        self.cov_alpha = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
        self.cov_beta = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))

        self.mean_alpha = gpytorch.means.ZeroMean()
        self.mean_beta = gpytorch.means.ConstantMean()
        self.mean_beta.initialize(constant=torch.tensor(b, requires_grad=False))
        self.mean_beta.raw_constant.requires_grad = False

        self.alpha_gp = GP(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood(), mean_module=self.mean_alpha, covar_module=self.cov_alpha)
        self.beta_gp = GP(train_x, train_y, gpytorch.likelihoods.GaussianLikelihood() , mean_module=self.mean_beta, covar_module=self.cov_beta)

        self.controller = controller

    def optimize(self, optim_steps:int, verbose:bool):
        return optimize_gp(self, optim_steps, verbose)
    
    def covar_alpha_beta(self, x:torch.Tensor):
        return self.cov_alpha(x) + self.U.transpose(0,1) * self.cov_beta(x) * self.U + self.Var

    def forward(self, x:torch.Tensor):

        mean_y = + self.train_u * self.mean_beta(x)
        #covar_y = self.cov_alpha(x) + self.U.transpose(0,1) * self.cov_beta(x) * self.U + self.Var #torch.eye(x.shape[0]) * 1e-6  #TODO
        covar_y = self.covar_alpha_beta(x)

        return gpytorch.distributions.MultivariateNormal(mean_y, covar_y)
    
    def alpha(self, _x:torch.Tensor):
        x = torch.tensor(_x).unsqueeze(0)
        # covar_y = self.cov_alpha(self.train_inputs[0]) + self.U.transpose(0,1) * self.cov_beta(self.train_inputs[0]) * self.U + torch.eye(self.train_inputs[0].shape[0]) * 1e-6
        covar_y = self.covar_alpha_beta(self.train_inputs[0])

        alpha = self.cov_alpha(self.train_inputs[0], x ).t() @ covar_y.to_dense().inverse() @ (self.train_targets - self.train_u * self.mean_beta(self.train_inputs[0]))
        return alpha.detach()
    
    def beta(self, _x:torch.Tensor, _u:torch.Tensor=0):
        x = torch.tensor(_x).unsqueeze(0)

        u = torch.tensor(self.controller(_x, _u))#.unsqueeze(0)
        # covar_y = self.cov_alpha(self.train_inputs[0]) + self.U.transpose(0,1) * self.cov_beta(self.train_inputs[0]) * self.U + torch.eye(self.train_inputs[0].shape[0]) * 1e-6
        covar_y = self.covar_alpha_beta(self.train_inputs[0])
        beta =  self.mean_beta(x) + u * self.cov_beta(self.train_inputs[0], x).t() @ self.U @  covar_y.to_dense().inverse() @ (self.train_targets - self.train_u * self.mean_beta(self.train_inputs[0]))
        return beta.detach()
    
    def get_nonlinear_fcts(self):
        return self.alpha, self.beta



#TODO
class MultiOutputGPModel(gpytorch.models.ExactGP):
    def __init__(self, _train_x:List[torch.Tensor], _train_u:List[torch.Tensor], _train_y:List[torch.Tensor], _likelihood:gpytorch.likelihoods.Likelihood, **kwargs):
        train_x = torch.cat(_train_x, dim=0)
        train_u = torch.cat(_train_u, dim=0)
        train_y = torch.cat(_train_y, dim=0)
        self.train_u = train_u

        likelihood = MultiOutputLikelihood()


        super().__init__(train_x, train_y, likelihood)

        self.alpha = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood(), mean_module=gpytorch.means.ZeroMean())


        self.mean_alpha = gpytorch.means.ConstantMean()
        self.mean_beta = gpytorch.means.ConstantMean()
        self.cov_alpha = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
        self.cov_beta = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))

    def forward(self, x):
        alpha_gp = gpytorch.distributions.MultivariateNormal(self.mean_alpha(x), self.cov_alpha(x))
        beta_gp = gpytorch.distributions.MultivariateNormal(self.mean_beta(x), self.cov_beta(x))
        return alpha_gp, beta_gp
    
    def optimize(self, optim_steps:int, verbose:bool):
        self.train()
        self.likelihood.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        for i in range(100):
            optimizer.zero_grad()
            f_pred, g_pred = self(self.train_inputs[0])
            output = self.likelihood(f_pred, g_pred, self.train_u)
            loss = -mll(output, self.train_targets.squeeze())
            loss.backward()
            optimizer.step()

    def get_nonlinear_fcts(self):
        def alpha(x):
            self.eval()
            with torch.no_grad():
                alpha_pred, _ = self(x)  # Only f(x)
            return alpha_pred.mean, alpha_pred.variance.sqrt()
            # alpha_gp = gpytorch.distributions.MultivariateNormal(self.mean_alpha(x), self.cov_alpha(x))
            # return alpha_gp.mean

        def beta(x, u):
            self.eval()
            with torch.no_grad():
                _, beta_pred = self(x)  # Only f(x)
            return beta_pred.mean, beta_pred.variance.sqrt()
            # beta_gp = gpytorch.distributions.MultivariateNormal(self.mean_beta(x), self.cov_beta(x))
            # return beta_gp.mean

        return alpha, beta

class MultiOutputLikelihood(gpytorch.likelihoods._GaussianLikelihoodBase):
    def __init__(self, base_likelihoood:gpytorch.likelihoods.Likelihood=None):
        noise_covar = gpytorch.likelihoods.noise_models.HomoskedasticNoise(
            # noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        super().__init__(noise_covar=noise_covar)
        self.noise = base_likelihoood if base_likelihoood is not None else gpytorch.likelihoods.GaussianLikelihood()
        # self.noise = gpytorch.likelihoods.GaussianLikelihood() #TODO: replace with heteroscedastic noise

    def forward(self, alpha_gp:gpytorch.distributions.Distribution, beta_gp:gpytorch.distributions.Distribution, u:torch.Tensor):
        mean_y = alpha_gp.mean + beta_gp.mean * u.squeeze(-1)
        cov_y = alpha_gp.covariance_matrix + (u @ u.T) * beta_gp.covariance_matrix
        return gpytorch.distributions.MultivariateNormal(mean_y, cov_y + torch.eye(len(mean_y)) * self.noise.noise) #TODO: noise_covar or noise
    

class Linearizing_Control_5(gpytorch.models.ExactGP):
    def __init__(self, x:torch.Tensor, u:torch.Tensor, y_ref:torch.Tensor, b:float, a:torch.Tensor, v:torch.Tensor, variance, **kwargs):
        train_x = torch.cat(x, dim=0)
        train_u = torch.cat(u, dim=0)
        train_y = torch.cat(y_ref, dim=0) # TODOL output is dim 3: create two masked channels before
        train_y = torch.stack([torch.full_like(train_y, torch.nan), torch.full_like(train_y, torch.nan), train_y], dim=-1)

        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, )# noise_constraint=gpytorch.constraints.GreaterThan(torch.tensor(1e-15))
        # task_noise = torch.full((3 * var[0].shape[0] * len(var),), float('nan'))
        # task_noise[2::3] = torch.cat(var, dim=0).squeeze()
        # task_noise = torch.cat(var, dim=0).squeeze().repeat_interleave(3)

        # likelihood = FixedTaskNoiseMultitaskLikelihood(num_tasks=3, noise=torch.tensor([1e-8,1e-8]), rank=3, has_task_noise=True, task_noise=task_noise)

        super().__init__(train_x,  train_y, likelihood)
        
        self.num_tasks = 3

        self.mean_module = FeedbackControl_Mean(b, a ,v)
        self.covar_module = FeedbackControl_Kernel(a, v)

        
        self.train_u = train_u

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def optimize(self, optim_steps:int, verbose:bool):
        return optimize_gp(self, optim_steps, verbose)
    
    def get_nonlinear_fcts(self):
        self.eval()
        self.likelihood.eval()
        def alpha(x):
            if len(x.shape) == 1:
                x = torch.tensor(x).unsqueeze(0)
            return self(torch.tensor(x)).mean[:, 0].unsqueeze(-1)

        def beta(x, u):
            if len(x.shape) == 1:
                x = torch.tensor(x).unsqueeze(0)
            return self(torch.tensor(x)).mean[:, 1].unsqueeze(-1)

        return alpha, beta
    
class VariationalGP(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
class CompositeModel(torch.nn.Module):
    def __init__(self, train_x:torch.Tensor, train_u:torch.Tensor, train_y:torch.Tensor, **kwargs):
        self.train_u = torch.cat(train_u, dim=0)
        self.train_targets = torch.cat(train_y, dim=0)
        self.train_inputs = [torch.cat(train_x, dim=0)]

        inducing_length = math.floor(train_x[0].shape[0] / 2)
        train_z = torch.cat([x[0:inducing_length] for x in train_x], dim=0)
        #FIXME: How to choose inducing points from state space?
        # Choose inducing points by sampling uniformly from the input space
        # Assuming the input space is bounded, define the bounds
        x_min, x_max = train_x[0][:, 0].min(), train_x[0][:, 0].max()
        y_min, y_max = train_x[0][:, 1].min(), train_x[0][:, 1].max()

        # Generate a grid of inducing points within the bounds
        l = 5
        inducing_points_x, inducing_points_y = torch.meshgrid(
            torch.linspace(x_min, x_max, l),
            torch.linspace(y_min, y_max, l)
        )
        inducing_points = torch.stack([inducing_points_x.flatten(), inducing_points_y.flatten()], dim=-1)

        super().__init__()

        self._alpha = VariationalGP(inducing_points)
        self._log_beta = VariationalGP(inducing_points)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def forward(self, x, u):
        alpha = self._alpha(x)
        log_beta = self._log_beta(x)
        beta = squared_gaussian(log_beta)
        
        mean = alpha.mean + beta.mean * u
        covar = alpha.covariance_matrix + u.unsqueeze(0)*beta.covariance_matrix * u.unsqueeze(1)
        y_pred = gpytorch.distributions.MultivariateNormal(mean, covar)
        return y_pred, alpha, beta
    
    def optimize(self, optim_steps:int, verbose:bool):
        self._alpha.train()
        self._log_beta.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self._log_beta, num_data=self.train_inputs[0].size(0))

        train_loss = []

        for i in range(optim_steps * 10):
            optimizer.zero_grad()
            y_pred , alpha, log_beta = self(self.train_inputs[0], self.train_u)
            loss = -mll(y_pred, self.train_targets)
            train_loss.append(loss)
            loss.backward()
            if i % 50 == 0 and verbose:
                print(f"Iter {i}, Loss: {loss.item():.4f}")
            optimizer.step()

    def get_nonlinear_fcts(self):
        self.eval()
        self.likelihood.eval()
        def alpha(x):
            if len(x.shape) == 1:
                x = torch.tensor(x).unsqueeze(0)
            return self._alpha(torch.tensor(x)).mean

        def beta(x, u):
            if len(x.shape) == 1:
                x = torch.tensor(x).unsqueeze(0)
            log_beta = self._log_beta(torch.tensor(x))
            _beta = squared_gaussian(log_beta)
            return _beta.mean
            # return torch.exp(log_beta.mean)
        
        return alpha, beta