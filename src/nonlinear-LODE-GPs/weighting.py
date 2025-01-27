import torch
import gpytorch
from gpytorch.kernels.kernel import sq_dist, dist

class Weighting_Function(gpytorch.Module):#gpytorch.Module
    def __init__(self, center:torch.Tensor, lengthscale:torch.Tensor):
        super(Weighting_Function, self).__init__()
        self.center = center
        #self.lengthscale = torch.nn.Parameter(torch.ones(1)*(44194))
        self.lengthscale = lengthscale


    def forward(self, x):
        center = self.center
        
        # x_ = x.div(self.lengthscale)
        # center_ = center.div(self.lengthscale)
        unitless_sq_dist = self.covar_dist(x, center, square_dist=True)
        # clone because inplace operations will mess with what's saved for backward
        covar_mat = unitless_sq_dist.div_(-2.0*self.lengthscale).exp_()
        return covar_mat
    
        # return RBFCovariance.apply(
        #     input,
        #     self.center,
        #     self.lengthscale,
        #     lambda input, center: self.covar_dist(input, center, square_dist=True, diag=False),
        # )
    def covar_dist(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        square_dist: bool = False,
        **params,
    ) -> torch.Tensor:
        

        x1_eq_x2 = torch.equal(x1, x2)
        dist_func = sq_dist if square_dist else dist
        return dist_func(x1, x2, x1_eq_x2)
    