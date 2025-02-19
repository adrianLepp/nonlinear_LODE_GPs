
from copy import deepcopy

import torch
from torch.nn import ModuleList

from gpytorch.means import Mean, MultitaskMean
import gpytorch


class Equilibrium_Mean(MultitaskMean):
    def __init__(self, mean_values:torch.Tensor, num_tasks:int, prior_variance=1e-8, mean_deviation=1e-12):
        mean_modules = []
        for i in range(len(mean_values)):
            mean_modules.append(gpytorch.means.ConstantMean(
                #constant_prior=gpytorch.priors.NormalPrior(mean_values[i],prior_variance),
                #constant_constraint=gpytorch.constraints.Interval(mean_values[i]-mean_deviation, mean_values[i]+mean_deviation)
            ))
            mean_modules[i].initialize(constant=torch.tensor(mean_values[i], requires_grad=False))
            mean_modules[i].raw_constant.requires_grad = False

        super().__init__(mean_modules, num_tasks)

class LODE_Mean(Mean):
    """
    
    """

    def __init__(self, multivariate_mean, V):
        """
        Args:
        
        """
        super(LODE_Mean, self).__init__()

        self.multivariate_mean = multivariate_mean
        self.V = V

    def forward(self, input):
        """
        
        """
        return self.V @ self.multivariate_mean(input)
    



    