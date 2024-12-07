
from copy import deepcopy

import torch
from torch.nn import ModuleList

from gpytorch.means import Mean

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