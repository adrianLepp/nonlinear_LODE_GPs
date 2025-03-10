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
import math

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
            train_y:List[torch.Tensor],
            likelihood,
            consecutive_training = True
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
            self.beta = GP(
                torch.cat(train_x[len_per_model::], dim=0), 
                torch.cat(train_y[len_per_model::], dim=0), 
                gpytorch.likelihoods.GaussianLikelihood(), 
                #mean_module=gpytorch.means.ZeroMean()
            )

            #self.beta = GP(train_x[1], train_y[1], gpytorch.likelihoods.GaussianLikelihood(), covar_module=gpytorch.kernels.ConstantKernel())    
            self.train_u = torch.cat(train_u[len_per_model::])
        else:
            self.train_x = torch.cat(train_x, dim=0)
            self.train_u = torch.cat(train_u, dim=0)
            self.train_y = torch.cat(train_y, dim=0)

            self.alpha = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood())#, mean_module=gpytorch.means.ZeroMean())
            self.beta = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood())#, covar_module=gpytorch.kernels.ConstantKernel())

    def optimize(self, training_iterations=100, verbose=False):
        if self.consecutive:
            self.optimize_consecutive(training_iterations, verbose)
        else:
            self.optimize_all(training_iterations, verbose)

    def optimize_all(self, training_iterations=100, verbose=False):
        self.alpha.train()
        self.alpha.likelihood.train()
        self.beta.train()
        self.beta.likelihood.train()

        # 2 Optimizers
        # optimizer_alpha = torch.optim.Adam(self.alpha.parameters(), lr=0.1)
        # optimizer_beta = torch.optim.Adam(self.beta.parameters(), lr=0.3)

        optimizer = torch.optim.Adam(list(self.alpha.parameters()) + list(self.beta.parameters()), lr=0.1)

        # 2 marginal log likelihoods
        mll_alpha = gpytorch.mlls.ExactMarginalLogLikelihood(self.alpha.likelihood, self.alpha)
        mll_beta = gpytorch.mlls.ExactMarginalLogLikelihood(self.beta.likelihood, self.beta)

        training_loss = [] 
        for i in range(training_iterations): 
            optimizer.zero_grad()
            # optimizer_alpha.zero_grad()
            # optimizer_beta.zero_grad()

            output_alpha = self.alpha(self.train_x)
            output_beta = self.beta(self.train_x)

            # 2 losses
            loss_alpha = -mll_alpha(output_alpha, self.train_y - output_beta.mean * self.train_u)
            loss_beta = -mll_beta(output_beta, self.train_y - output_alpha.mean)
            loss = loss_alpha + loss_beta
            # loss_alpha.backward()
            # loss_beta.backward()
            loss.backward()

            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, (loss.item())))
            
            # 2 optimizers
            # optimizer_alpha.step()
            # optimizer_beta.step()
            optimizer.step()

            # training_loss.append(loss.item())      


    def optimize_consecutive(self, training_iterations=100, verbose=False):
        self.optimize_alpha(training_iterations, verbose)
        self.alpha.eval()
        self.alpha.likelihood.eval()

        train_targets_beta = (self.beta.train_targets - self.alpha(self.beta.train_inputs[0]).mean.detach().clone())/ self.train_u
        self.beta.set_train_data(self.beta.train_inputs[0], train_targets_beta.detach())

        self.optimize_beta(training_iterations, verbose)

        self.beta.eval()
        self.beta.likelihood.eval()

    def optimize_alpha(self,training_iterations:int=100, verbose=False):
        self.alpha.train()
        self.alpha.likelihood.train()
        optimizer = torch.optim.Adam(self.alpha.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.alpha.likelihood, self.alpha)
        training_loss = [] 
        for i in range(training_iterations): 
            optimizer.zero_grad()
            output = self.alpha(self.alpha.train_inputs[0])
            loss = -mll(output, self.alpha.train_targets)
            loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
            training_loss.append(loss.item())

    def optimize_beta(self,training_iterations:int=100, verbose=False):
        self.beta.train()
        self.beta.likelihood.train()
        optimizer = torch.optim.Adam(self.beta.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.beta.likelihood, self.beta)
        training_loss = [] 
        for i in range(training_iterations): 
            optimizer.zero_grad()
            output = self.beta(self.beta.train_inputs[0]) #self.beta(self.beta.train_inputs[0])
            #loss = -mll(output, (self.beta.train_targets - self.alpha(self.beta.train_inputs[0]).mean)/ self.train_u)
            loss = -mll(output, self.beta.train_targets) 
            loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
            training_loss.append(loss.item())
    

    def y_ref(self, x:torch.Tensor, u:torch.Tensor):
        output = self(x, u)
        return output.mean

        # alpha = self.alpha.likelihood(self.alpha(x)).mean
        # beta = self.beta.likelihood(self.beta(x)).mean

        # mean_x =  alpha + beta * u.squeeze()

        # return mean_x
    
    def forward(self, x:torch.Tensor, u:torch.Tensor):
        return y_ref_from_alpha_beta(self.alpha(x), self.beta(x), u)


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