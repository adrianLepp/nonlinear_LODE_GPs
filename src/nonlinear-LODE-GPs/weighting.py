import gpytorch.constraints
import torch
import gpytorch
from gpytorch.kernels.kernel import sq_dist, dist

class Weighting_Function(gpytorch.Module):#gpytorch.Module
    #def __init__(self, center:torch.Tensor, lengthscale_prior:torch.Tensor):
    def __init__(self, center:torch.Tensor, length_prior=None, length_constraint=None,):
        super(Weighting_Function, self).__init__()
        self.center = center
        #self.lengthscale = torch.nn.Parameter(torch.ones(1)*(44194))
        self.lengthscale = length_prior


        self.register_parameter(
            #name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(1, 1))
        )
        
        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = gpytorch.constraints.Positive()

        # register the constraint
        self.register_constraint("raw_length", length_constraint)
        
        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if length_prior is not None:
            self.register_prior(
                "length_prior",
                length_prior,
                lambda m: m.length,
                lambda m, v : m._set_length(v),
            )

    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    def forward(self, x):
        center = self.center
        
        # x_ = x.div(self.lengthscale)
        # center_ = center.div(self.lengthscale)
        unitless_sq_dist = self.covar_dist(x, center, square_dist=True)
        # clone because inplace operations will mess with what's saved for backward
        covar_mat = unitless_sq_dist.div_(-2.0*self.length).exp_()
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
    