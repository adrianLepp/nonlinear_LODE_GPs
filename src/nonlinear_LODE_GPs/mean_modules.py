
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


class Controller_Mean(Mean):
    def __init__(self, sub_mean:Mean, a:torch.Tensor, v:torch.Tensor):
        super().__init__()
        self.a = a
        self.v = v
        self.sub_mean = sub_mean

    def control_law(self, x:torch.Tensor, y_ref=0):
        return - x @ self.a # + self.v * y_ref
    
    def forward(self, x):
        b = self.sub_mean(x)
        u = self.control_law(x).squeeze(1)
        mean = b * u
        return mean


class FeedbackControl_Mean(MultitaskMean):
    def __init__(self, b, a:torch.Tensor, v:torch.Tensor):
        mean_beta = gpytorch.means.ConstantMean()
        # mean_beta.initialize(constant=torch.tensor(b, requires_grad=False))
        # mean_beta.raw_constant.requires_grad = False
        mean_modules = [
            gpytorch.means.ZeroMean(),
            mean_beta,
            Controller_Mean(mean_beta, a, v)
        ]

        super().__init__(mean_modules, 3)
    