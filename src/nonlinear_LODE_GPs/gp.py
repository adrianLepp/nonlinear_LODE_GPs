import gpytorch
import torch

class GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    


class Linearizing_Control(gpytorch.models.ExactGP):
    def __init__(self, train_x:torch.Tensor, train_y:torch.Tensor, likelihood):
        super().__init__(train_x,  train_y, likelihood)


        self.a_mean_module = gpytorch.means.ConstantMean()
        self.a_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

        self.b_mean_module = gpytorch.means.ConstantMean()
        self.b_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x:torch.Tensor):
        u = x[:,-1] .unsqueeze(0)
        mean_x =  self.b_mean_module(x[:,0:-1])* u.squeeze() + self.a_mean_module(x[:,0:-1])
        covar_x = u * self.b_covar_module(x[:,0:-1]) *u.transpose(0,1) + self.a_covar_module(x[:,0:-1])
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
