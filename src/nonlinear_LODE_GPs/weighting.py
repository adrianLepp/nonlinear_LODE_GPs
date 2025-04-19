import gpytorch.constraints
import torch
import gpytorch
from gpytorch.kernels.kernel import sq_dist, dist

class Gaussian_Weight(gpytorch.Module):
    def __init__(self, center:torch.Tensor, scale_prior=None, scale_constraint=None, shared_weightscale=False):
        super(Gaussian_Weight, self).__init__()
        self.center = center

        self.shared_weightscale = shared_weightscale
        if not shared_weightscale:
            self.register_parameter(
            #name='raw_scale', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
            name='raw_scale', parameter=torch.nn.Parameter(torch.ones(1, 1))
            )
        
            if scale_constraint is None:
                scale_constraint = gpytorch.constraints.Positive()

            if scale_constraint is not None:
                self.register_constraint("raw_scale", scale_constraint)
            
            if scale_prior is not None:
                self.register_prior(
                    "scale_prior",
                    scale_prior,
                    lambda m: m.scale,
                    lambda m, v : m._set_scale(v),
                )

    @property
    def scale(self):
        if hasattr(self, "raw_scale_constraint"):
            return self.raw_scale_constraint.transform(self.raw_scale)
        return self.raw_scale

    @scale.setter
    def scale(self, value):
        return self._set_scale(value)

    def _set_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_scale)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        if hasattr(self, "raw_scale_constraint"):
            self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))
        else:
            self.initialize(raw_scale=value)

    def forward(self, x:gpytorch.distributions.Distribution, scale=None):
        center = self.center
        if not self.shared_weightscale:
            scale = self.scale

        if isinstance(x, gpytorch.distributions.Distribution):
            x = x.mean
        
        unitless_sq_dist = self.covar_dist(x, center, square_dist=True)
        # clone because inplace operations will mess with what's saved for backward
        covar_mat = unitless_sq_dist.div_(-2.0*scale).exp_()
        return covar_mat
    
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
    

class Constant_Weight(gpytorch.Module):#gpytorch.Module
    def __init__(self):
        super(Constant_Weight, self).__init__()
        self.register_parameter(
            name='raw_weight', parameter=torch.nn.Parameter(torch.ones(1, 1))
        )
        self.register_constraint("raw_weight", gpytorch.constraints.Positive())

    @property
    def weight(self):
        return self.raw_weight_constraint.transform(self.raw_weight)
        
    @weight.setter
    def weight(self, value):
        return self._set_weight(value)

    def _set_weight(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weight)
        self.initialize(raw_weight=self.raw_weight_constraint.inverse_transform(value))
        

    def forward(self, x:gpytorch.distributions.Distribution, *args, **kwargs):
        if isinstance(x, gpytorch.distributions.Distribution):
            x = x.mean
        return torch.ones((x.shape[0],1)) * self.weight
    
class KL_Divergence_Weight(gpytorch.Module):
    def __init__(self, center:torch.Tensor, scale_prior=None, scale_constraint=None, shared_weightscale=False):
        super(KL_Divergence_Weight, self).__init__()
        self.ignore_control = False #FIXME
        self.center = center
        self.num_tasks = 3 #FIXME
        # self.scale = 0.001 #TODO This should be learned. 0.5 for localgp

        self.shared_weightscale = shared_weightscale
        if not shared_weightscale:


            self.register_parameter(
                name='raw_scale', parameter=torch.nn.Parameter(torch.ones(1, 1))
            )
            
            if scale_constraint is not None:
                self.register_constraint("raw_scale", scale_constraint)
            
            if scale_prior is not None:
                self.register_prior(
                    "scale_prior",
                    scale_prior,
                    lambda m: m.scale,
                    lambda m, v : m._set_scale(v),
                )

    @property
    def scale(self):
        if hasattr(self, "raw_scale_constraint"):
            return self.raw_scale_constraint.transform(self.raw_scale)
        return self.raw_scale

    @scale.setter
    def scale(self, value):
        return self._set_scale(value)

    def _set_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_scale)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        if hasattr(self, "raw_scale_constraint"):
            self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))
        else:
            self.initialize(raw_scale=value)

    def forward_without_control(self, dist:gpytorch.distributions.Distribution, scale):
        center = self.center[:,:-1] 
        N = dist.mean.shape[0]
        num_tasks = 3
        reduced_covar = dist.covariance_matrix.reshape(N,num_tasks,N,num_tasks)[:,:-1,:,:-1].reshape(N*(num_tasks-1),N*(num_tasks-1)) + torch.eye(N*(num_tasks-1))*1e-6 #FIXME
        reduced_mean = dist.mean[:,:-1]
        reduced_dist = gpytorch.distributions.MultitaskMultivariateNormal(reduced_mean, reduced_covar)
        center_dist = gpytorch.distributions.MultitaskMultivariateNormal(center.tile(N,1), reduced_covar)
        divergence = torch.tensor([torch.distributions.kl_divergence(reduced_dist[i,:], center_dist[i,:]) for i in range(N)])
        weight = 1 / (scale.squeeze()* divergence + 1).unsqueeze(1)
        return weight

    def forward(self, dist:gpytorch.distributions.Distribution, scale=None):
        if not isinstance(dist, gpytorch.distributions.Distribution):
            raise ValueError('Input must be a gpytorch distribution')
        
        if not self.shared_weightscale:
            scale = self.scale
        
        if self.ignore_control:
            return self.forward_without_control(dist, scale)
        
        center = self.center
        N = dist.mean.shape[0]
        new_dist = gpytorch.distributions.MultitaskMultivariateNormal(dist.mean, torch.diag(torch.abs(torch.diag(dist.covariance_matrix)))) 
        # center_dist = gpytorch.distributions.MultitaskMultivariateNormal(center.tile(N,1), dist.covariance_matrix)
        center_dist = gpytorch.distributions.MultitaskMultivariateNormal(center, torch.diag(torch.tensor([1e-1,1e-1,1e-4])))
        divergence = torch.tensor([torch.distributions.kl_divergence(new_dist[i,:], center_dist) for i in range(N)])

        # import matplotlib.pyplot as plt

        
        # plt.figure(figsize=(10, 5))
        # plt.plot(center_dist.mean.detach().numpy(), label="center_dist")
        # plt.plot(dist.mean[0, :].detach().numpy(), label="dist[0,:]")
        
        # plt.plot(dist.mean[-1, :].detach().numpy(), label="dist[-1,:]")
        # plt.legend()
        # plt.show()
    
        weight = 1 / (scale.squeeze()* divergence + 1).unsqueeze(1)
        # weight = 1 / ( divergence + 1).unsqueeze(1)
        return weight
    

class Epanechnikov_Weight(gpytorch.Module):
    def __init__(self, center:torch.Tensor, length_prior=None, length_constraint=None,):
        super(Epanechnikov_Weight, self).__init__()
        self.center = center
        self.lengthscale = length_prior


        self.register_parameter(
            #name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1))
            name='raw_length', parameter=torch.nn.Parameter(torch.ones(1, 1, requires_grad=False))
        )
        
        # if length_constraint is None:
        #     length_constraint = gpytorch.constraints.Positive()

        if length_constraint is not None:
            self.register_constraint("raw_length", length_constraint)
        
        if length_prior is not None:
            self.register_prior(
                "length_prior",
                length_prior,
                lambda m: m.length,
                lambda m, v : m._set_length(v),
            )

    @property
    def length(self):
        if hasattr(self, "raw_length_constraint"):
            return self.raw_length_constraint.transform(self.raw_length)
        return self.raw_length
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        if hasattr(self, "raw_length_constraint"):
            self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))
        else:
            self.initialize(raw_length=value)

    def forward(self, x:gpytorch.distributions.Distribution):
        center = self.center

        if isinstance(x, gpytorch.distributions.Distribution):
            x = x.mean
        
        const = torch.tensor(0.03)
        factor = torch.sqrt(self.length)
        unitless_sq_dist = self.covar_dist(x, center, square_dist=True)
        mask = torch.where(unitless_sq_dist > self.length, torch.tensor(0.0, device=unitless_sq_dist.device), torch.tensor(1.0, device=unitless_sq_dist.device))

        epachnikov = 3 / (4 * factor) * (1-unitless_sq_dist.div(self.length))
        return epachnikov * mask
    
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
    

class Mahalanobis_Distance(gpytorch.Module):
    def __init__(self, center:torch.Tensor, scale_prior=None, scale_constraint=None, shared_weightscale=False):
        super(Mahalanobis_Distance, self).__init__()
        self.center = center
        self.num_tasks = 3 #FIXME

        self.shared_weightscale = shared_weightscale
        if not shared_weightscale:


            self.register_parameter(
                name='raw_scale', parameter=torch.nn.Parameter(torch.ones(1, 1)*1200)
            )
            
            if scale_constraint is not None:
                self.register_constraint("raw_scale", scale_constraint)
            
            if scale_prior is not None:
                self.register_prior(
                    "scale_prior",
                    scale_prior,
                    lambda m: m.scale,
                    lambda m, v : m._set_scale(v),
                )

    @property
    def scale(self):
        if hasattr(self, "raw_scale_constraint"):
            return self.raw_scale_constraint.transform(self.raw_scale)
        return self.raw_scale

    @scale.setter
    def scale(self, value):
        return self._set_scale(value)

    def _set_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_scale)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        if hasattr(self, "raw_scale_constraint"):
            self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))
        else:
            self.initialize(raw_scale=value)

    def mahalanobis_dist(self, distribution:gpytorch.distributions.Distribution, sqrt=True):
        distance = distribution.mean.detach() - self.center
        
        md = (distance @ distribution.covariance_matrix.detach().inverse() @ distance.t()).squeeze()
        if sqrt:
            return torch.sqrt(md)
        else:
            return md

    def forward(self, dist:gpytorch.distributions.Distribution, scale=None):
        if not isinstance(dist, gpytorch.distributions.Distribution):
            raise ValueError('Input must be a gpytorch distribution')
        
        if not self.shared_weightscale:
            scale = self.scale
        
        N = dist.mean.shape[0]

        # md = torch.stack([self.mahalanobis_dist(dist[i, :], sqrt = False) for i in range(N)])
        md = torch.stack([self.mahalanobis_dist(dist[i, :], sqrt = True) for i in range(N)])
        # weights = torch.exp(-0.5 * md.div(self.scale))
        weights = md.div(self.scale**2)
        return weights.t()
        # weight = torch.stack([self.mahalanobis_dist(dist[i, :]) for i in range(N)])
        
        # return weight.div(self.scale).t()