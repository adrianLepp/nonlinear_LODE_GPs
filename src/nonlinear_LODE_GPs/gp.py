import gpytorch
import torch

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
            train_x_alpha:torch.Tensor, 
            train_u_alpha:torch.Tensor, 
            train_y_alpha:torch.Tensor,
            train_x_beta:torch.Tensor, 
            train_u_beta:torch.Tensor, 
            train_y_beta:torch.Tensor, 
            likelihood,
            # system,
            # a_0,
            # a_1,
            # v
        ):
        super().__init__()
        self.train_x_alpha = train_x_alpha
        self.train_u_alpha = train_u_alpha
        self.train_y_alpha = train_y_alpha
        self.train_x_beta = train_x_beta
        self.train_u_beta = train_u_beta
        self.train_y_beta = train_y_beta

        self.likelihood = likelihood

        self.alpha = GP(train_x_alpha, train_y_alpha, likelihood, mean_module=gpytorch.means.ZeroMean())
        self.beta = GP(train_x_beta, train_y_beta, likelihood, covar_module=gpytorch.kernels.ConstantKernel()) #


        # self.system  = system
        # self.a_0 = a_0
        # self.a_1 = a_1
        # self.v = v

    def optimize(self, training_iterations=100, verbose=False):
        self.optimize_alpha(training_iterations, verbose)
        self.alpha.eval()
        self.alpha.likelihood.eval()

        self.optimize_beta(training_iterations*10, verbose)

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
            loss = -mll(output, self.beta.train_targets - self.alpha(self.beta.train_inputs[0]).mean)
            loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()
            training_loss.append(loss.item())
    

    def y_ref(self, x:torch.Tensor, u:torch.Tensor):
        alpha = self.likelihood(self.alpha(x)).mean
        beta = self.likelihood(self.beta(x)).mean

        mean_x =  alpha + beta * u.squeeze()

        return mean_x
    
    # def u(self, x, y_ref, u):
    #     alpha = self.likelihood(self.alpha(x)).mean
    #     beta = self.likelihood(self.beta(x)).mean

    #     u = (y_ref - alpha) / beta

    #     return u
