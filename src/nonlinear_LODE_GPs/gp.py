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
            likelihood,
            consecutive_training = True,
            **kwargs
        ):
        super().__init__()

        self.likelihood = likelihood

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
            
            self._alpha = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood(), gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)))#, mean_module=gpytorch.means.ZeroMean())
            self._beta = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood(), gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)))#, covar_module=gpytorch.kernels.ConstantKernel())

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
            output_beta = self._beta(self.train_x)
            output = y_ref_from_alpha_beta(output_alpha, output_beta, self.train_u)    

            loss = -mll(output, self.train_y )
            loss.backward()

            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, (loss.item())))
            
            optimizer.step()

            training_loss.append(loss.item())

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
        return y_ref_from_alpha_beta(self._alpha(x), self._beta(x), u)
    
    def get_nonlinear_fcts(self):
        self._beta.eval()
        self._alpha.eval()
    
        def alpha(x):
            return self._alpha(torch.tensor(x).unsqueeze(0)).mean.clone().detach().numpy()
            # with torch.no_grad:
            #       return control_gp.alpha(torch.tensor(x).unsqueeze(0)).mean.numpy()

        def beta(x,u):
            return self._beta(torch.tensor(x).unsqueeze(0)).mean.clone().detach().numpy()
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

class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))
    

class LinearizingContorl_DeepGP(DeepGP):

    def __init__(self, train_x_shape):
        hidden_layer = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=2,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        # self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()

    def forward(self, inputs, u):
        hidden_rep1 = self.hidden_layer(inputs)

        #TODO
        # output = self.last_layer(hidden_rep1)
        # return output
        if u is None:
            u = self.train_u

        return y_ref_from_alpha_beta(hidden_rep1[0], hidden_rep1[1], u)
        mean_x =  hidden_rep1[0].mean + hidden_rep1[1].mean * u
        covar_x = u.unsqueeze(0) * hidden_rep1[1].covariance_matrix * u.unsqueeze(1) + hidden_rep1[0].covariance_matrix
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)#torch.eye(covar_x.shape[0])
    
    def y_ref_from_alpha_beta(self, alpha:gpytorch.distributions.Distribution, beta:gpytorch.distributions.Distribution, u:torch.Tensor):
        mean_x =  alpha.mean + beta.mean * u
        covar_x = u.unsqueeze(0) * beta.covariance_matrix * u.unsqueeze(1) + alpha.covariance_matrix
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x + torch.eye(covar_x.shape[0]) * 1e-8)

    # def predict(self, test_loader):
    #     with torch.no_grad():
    #         mus = []
    #         variances = []
    #         lls = []
    #         for x_batch, y_batch in test_loader:
    #             preds = self.likelihood(self(x_batch, u_batch))
    #             mus.append(preds.mean)
    #             variances.append(preds.variance)
    #             lls.append(self.likelihood.log_marginal(y_batch, self(x_batch, u_batch)))

    #     return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
    
    # def optimize(self, training_iterations=100, verbose=False):
    #     num_samples = 10
    #     optimizer = torch.optim.Adam([
    #         {'params': self.parameters()},
    #     ], lr=0.01)
    #     mll = DeepApproximateMLL(VariationalELBO(self.likelihood, self, train_x.shape[-2]))

    #     epochs_iter = tqdm.notebook.tqdm(range(training_iterations), desc="Epoch")
    #     for i in epochs_iter:
    #         # Within each iteration, we will go over each minibatch of data
    #         minibatch_iter = tqdm.notebook.tqdm(train_loader, desc="Minibatch", leave=False)
    #         for x_batch, y_batch in minibatch_iter:
    #             with gpytorch.settings.num_likelihood_samples(num_samples):
    #                 optimizer.zero_grad()
    #                 output = self(x_batch, u_batch)
    #                 loss = -mll(output, y_batch)
    #                 loss.backward()
    #                 optimizer.step()

    #                 minibatch_iter.set_postfix(loss=loss.item())


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
