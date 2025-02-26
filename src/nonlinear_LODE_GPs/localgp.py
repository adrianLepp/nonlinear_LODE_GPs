import gpytorch 
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from nonlinear_LODE_GPs.kernels import *

import torch
from nonlinear_LODE_GPs.mean_modules import *
from nonlinear_LODE_GPs.weighting import *
from nonlinear_LODE_GPs.lodegp import *

DEBUG = False

import matplotlib.pyplot as plt
def plot_weights(x, weights, title="Weighting Function"):
    plt.figure(figsize=(12, 6))
    if isinstance(weights, list):
        for i, weight in enumerate(weights):
            plt.plot(x, weight.detach().numpy(), label=f'Weight {i}')
        plt.legend()
    else:
        plt.plot(x, weights.detach().numpy())
    plt.xlabel("x")
    plt.ylabel("Weight")
    plt.title(title)
    plt.show()


class Sum_LODEGP(gpytorch.models.ExactGP):
    def __init__(
            self, 
            train_x:torch.Tensor, 
            train_y:torch.Tensor, 
            likelihood:gpytorch.likelihoods.Likelihood, 
            num_tasks:int, 
            A_list:List, 
            equilibrium_list:List[torch.Tensor], 
            center_list:List[torch.Tensor], 
            weight_lengthscale:torch.Tensor, 
            output_distance=False,
            additive_kernel=False,
            pre_model=None,
            Weight_Model=Gaussian_Weight
        ):
        if output_distance is True and pre_model is None:
            raise ValueError("Output distance is True but no pre_model is given.")
        super(Sum_LODEGP, self).__init__(train_x, train_y, likelihood)

        self.num_tasks = num_tasks
        self.pre_model = pre_model
        for p in self.pre_model.parameters():
            p.requires_grad = False
        self.output_distance = output_distance 
        
        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }

        means = []
        kernels = []
        # _kernels = []
        w_functions = []
        for i in range(len(A_list)):
            kernels.append(LODE_Kernel(A_list[i], self.common_terms))
            means.append(Equilibrium_Mean(equilibrium_list[i], num_tasks))
            # w_fcts[i].initialize(length=torch.tensor(weight_lengthscale, requires_grad=True))#TODO
            
            w_functions.append(Weight_Model(center_list[i])) #TODO  Constant_Weight()
            w_functions[i].length = weight_lengthscale
            w_functions[i].initialize(raw_length=torch.tensor(weight_lengthscale, requires_grad=False))
            # w_functions[i].initialize(weight=torch.tensor(1/len(A_list), requires_grad=False))

        self.mean_module = Global_Mean(means, w_functions, num_tasks,output_distance)
        self.covar_module = Global_Kernel(kernels, w_functions, num_tasks,output_distance, additive_kernel)

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()

        with torch.no_grad():
            self.pre_model.eval()
            self.pre_model.likelihood.eval()
            pre_estimate = self.pre_model(X) if self.output_distance else None

        mean_x = self.mean_module(X, out=pre_estimate)
        covar_x = self.covar_module(X, out=pre_estimate, common_terms=self.common_terms)

        # if X % 10 == 0:
        # if DEBUG:
        #     plot_weights(X, mean_x[:,0], 'Global Mean')
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class Global_Mean(Mean):
    def __init__(self, local_means:List[Mean], weight_functions:List[Gaussian_Weight], num_tasks:int, output_distance:bool=False):
        super(Global_Mean, self).__init__()
        self.num_tasks = num_tasks
        self.local_means = ModuleList(local_means)
        self.weight_functions = ModuleList(weight_functions)
        self.output_distance = output_distance

    def forward(self, x:torch.Tensor, **params):
        mean_list = [local_mean(x) for local_mean in self.local_means]
        
        # distance_measure =  mean_list if self.output_distance else [x.clone() for _ in range(len(mean_list))] 
        distance_measure =  params["out"] if self.output_distance else x.clone() 

        weight_list = [self.weight_functions[i](distance_measure) for i in range(len(self.weight_functions))]

        mean = sum(mean_list[i] * weight_list[i] for i in range(len(mean_list)))/sum(weight_list)

        if DEBUG:
            plot_weights(x, weight_list)
            plot_weights(x, [mean_list[i][:,0] for i in range(len(mean_list))], 'Local Means')
            plot_weights(x, sum(weight_list), 'Sum of Weights')

        return mean
class Global_Kernel(Kernel):
    def __init__(self, local_kernels:List[Kernel], weight_functions:List[Gaussian_Weight], num_tasks, output_distance=False, additive_kernel=False):
        super(Global_Kernel, self).__init__(active_dims=None)
        self.num_tasks = num_tasks
        self.local_kernels = ModuleList(local_kernels)
        self.weight_functions = ModuleList(weight_functions)
        self.output_distance = output_distance

        if additive_kernel:
            self.additive_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MultitaskKernel(
                    gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=0
                ))

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

    def forward(self, x1, x2, diag=False, **params):
        if self.output_distance is False:
            distance_measure_1 = x1
            distance_measure_2 = x2
        else:
            out = params["out"]
            x1_eq_x2 = torch.equal(x1, x2)
            if (x1_eq_x2):
                distance_measure_1 = distance_measure_2 = out
            else:
                length_1 = x1.size(0)
                length_2 = x2.size(0)
                distance_measure_1 = out[:length_1]
                distance_measure_2 = out[-length_2:]

        covariances = [local_kernel(x1, x2, diag=False, **params) for local_kernel in self.local_kernels]
        
        # weight_matrices_1 = [torch.tile(weighting_function(distance_measure_1),(self.num_tasks,1)) for weighting_function in self.weight_functions]
        # weight_matrices_2 = [torch.tile(weighting_function(distance_measure_2),(self.num_tasks,1)) for weighting_function in self.weight_functions]
        weights_extended_1 = [weight_function(distance_measure_1).repeat_interleave(self.num_tasks).unsqueeze(1) for weight_function in self.weight_functions]
        weights_extended_2 = [weight_function(distance_measure_2).repeat_interleave(self.num_tasks).unsqueeze(0) for weight_function in self.weight_functions] 

        # weight = sum((weighting_function(distance_measure_1) * weighting_function(distance_measure_2).t())  for weighting_function in self.weight_functions)
        # weight_matrix = torch.tile(weight,(self.num_tasks,self.num_tasks))
        weight_matrix = sum(weights_extended_1[i] * weights_extended_2[i]  for i in range(len(weights_extended_1)))

        # covar = sum(weight_matrices_1[i] * covariances[i] * weight_matrices_2[i].t()  for i in range(len(covariances))) / weight_matrix
        covar = sum(weights_extended_1[i] * covariances[i] * weights_extended_2[i]  for i in range(len(covariances))) / weight_matrix 
        if self.additive_kernel is not None:
            covar = covar + self.additive_kernel(x1, x2, diag=False, **params)

        return covar