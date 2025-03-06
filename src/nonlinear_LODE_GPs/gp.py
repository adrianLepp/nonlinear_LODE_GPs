import gpytorch
import torch
from typing import List

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                  covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
                  mean_module = gpytorch.means.ConstantMean()
                  ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module #Constant Mean
        self.covar_module = covar_module

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

        if consecutive_training:
            if len(train_x) != 2:
                raise Warning('Consecutive training takes two training sets')
            self.alpha = GP(train_x[0], train_y[0], gpytorch.likelihoods.GaussianLikelihood(), mean_module=gpytorch.means.ZeroMean())
            self.beta = GP(train_x[1], train_y[1], gpytorch.likelihoods.GaussianLikelihood(), covar_module=gpytorch.kernels.ConstantKernel())    
            self.train_u = train_u
        else:
            self.train_x = torch.cat(train_x, dim=0)
            self.train_u = torch.cat(train_u, dim=0)
            self.train_y = torch.cat(train_y, dim=0)

            self.alpha = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood(), mean_module=gpytorch.means.ZeroMean())
            self.beta = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood(), covar_module=gpytorch.kernels.ConstantKernel())

    def optimize_all(self, training_iterations=100, verbose=False):
        self.alpha.train()
        self.alpha.likelihood.train()
        self.beta.train()
        self.beta.likelihood.train()

        optimizer_alpha = torch.optim.Adam(self.alpha.parameters(), lr=0.1)
        optimizer_beta = torch.optim.Adam(self.beta.parameters(), lr=0.3)
        mll_alpha = gpytorch.mlls.ExactMarginalLogLikelihood(self.alpha.likelihood, self.alpha)
        mll_beta = gpytorch.mlls.ExactMarginalLogLikelihood(self.beta.likelihood, self.beta)

        training_loss = [] 
        for i in range(training_iterations): 
            optimizer_alpha.zero_grad()
            output_alpha = self.alpha(self.train_x)

            optimizer_beta.zero_grad()
            output_beta = self.beta(self.train_x)


            loss_alpha = -mll_alpha(output_alpha, self.train_y - output_beta.mean * self.train_u)
            loss_beta = -mll_beta(output_beta, self.train_y - output_alpha.mean)
            loss_alpha.backward()
            loss_beta.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, (loss_alpha.item() + loss_beta.item())/2))
            optimizer_alpha.step()
            optimizer_beta.step()
            # training_loss.append(loss.item())      


    def optimize_consecutive(self, training_iterations=100, verbose=False):
        self.optimize_alpha(training_iterations, verbose)
        self.alpha.eval()
        self.alpha.likelihood.eval()

        self.optimize_beta(training_iterations * 2, verbose)

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
            loss = -mll(output, self.beta.train_targets - self.alpha(self.beta.train_inputs[0]).mean) #FIXME train_u
            loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
            training_loss.append(loss.item())
    

    def y_ref(self, x:torch.Tensor, u:torch.Tensor):
        alpha = self.alpha.likelihood(self.alpha(x)).mean
        beta = self.beta.likelihood(self.beta(x)).mean

        mean_x =  alpha + beta * u.squeeze()

        return mean_x
    
    # def u(self, x, y_ref, u):
    #     alpha = self.likelihood(self.alpha(x)).mean
    #     beta = self.likelihood(self.beta(x)).mean

    #     u = (y_ref - alpha) / beta

    #     return u


class Linearizing_Control_3(torch.nn.Module):
    def __init__(
            self, 
            train_x_alpha:torch.Tensor, 
            train_u_alpha:torch.Tensor, 
            train_y_alpha:torch.Tensor,
            train_x_beta:torch.Tensor, 
            train_u_beta:torch.Tensor, 
            train_y_beta:torch.Tensor, 
        ):
        super().__init__()
        self.train_x_alpha = train_x_alpha
        self.train_u_alpha = train_u_alpha
        self.train_y_alpha = train_y_alpha
        self.train_x_beta = train_x_beta
        self.train_u_beta = train_u_beta
        self.train_y_beta = train_y_beta

        self.train_u = torch.cat((train_u_alpha, train_u_beta), dim=0)
        self.train_x = torch.cat((train_x_alpha,train_x_beta), dim=0)
        self.train_y = torch.cat((train_y_alpha, train_y_beta), dim=0)
        # self.train_u = train_u_alpha
        # self.train_x = train_x_alpha
        # self.train_y = train_y_alpha

        # self.likelihood = likelihood

        self.alpha = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood(), mean_module=gpytorch.means.ZeroMean())
        self.beta = GP(self.train_x, self.train_y, gpytorch.likelihoods.GaussianLikelihood(), covar_module=gpytorch.kernels.ConstantKernel())
        self.model = gpytorch.models.IndependentModelList(self.alpha, self.beta)
        self.likelihood = gpytorch.likelihoods.LikelihoodList(self.alpha.likelihood, self.beta.likelihood)

    def optimize(self, training_iterations=100, verbose=False):
        self.model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.SumMarginalLogLikelihood(self.likelihood, self.model)
        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        training_loss = []

        for i in range(training_iterations):
        
            optimizer.zero_grad()
            output = self.model(*self.model.train_inputs)#FIXME: [0]??? 
            y_ref = self.y_ref_from_alpha_beta(*output, self.train_u)
            #output.mean[0] + output.mean[1] * self.model.train_inputs[:,-1] #FIXME: u is sth different
            loss = -mll([y_ref, y_ref], self.model.train_targets)
            loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

            training_loss.append(loss.item())

    def y_ref_from_alpha_beta(self, alpha:gpytorch.distributions.Distribution, beta:gpytorch.distributions.Distribution, u:torch.Tensor):
        mean_x =  alpha.mean + beta.mean * u
        covar_x = u.unsqueeze(0) * beta.covariance_matrix * u.unsqueeze(1) + alpha.covariance_matrix
        return gpytorch.distributions.MultivariateNormal(mean_x, torch.eye(covar_x.shape[0]))#FIXME

    def forward(self, x:torch.Tensor, u:torch.Tensor):
        output = self.model(x, x)
        y_ref:gpytorch.distributions.MultivariateNormal = self.y_ref_from_alpha_beta(*output, u)
        return y_ref
        # alpha = output[0]
        # beta = output[1]
        # mean_x =  alpha.mean + beta.mean * u
        # covar_x = u * beta.covar_matrix * u.t() + alpha.covar_matrix
        # return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)