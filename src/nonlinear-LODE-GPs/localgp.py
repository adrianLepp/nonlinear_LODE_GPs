import gpytorch 
from gpytorch.kernels import Kernel
from gpytorch.means import Mean
from kernels import *

import torch
from mean_modules import *
from weighting import *
from lodegp import *

DEBUG = False


class Sum_LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks:int, A_list:List, equilibrium_list:List[torch.Tensor], center_list:List[torch.Tensor], weight_lengthscale:torch.Tensor, output_distance=False):
        self.num_tasks = num_tasks
        super(Sum_LODEGP, self).__init__(train_x, train_y, likelihood)
        
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
            #means.append(Local_Mean(equilibrium_list[i], num_tasks, center_list[i], weight_lengthscale, output_distance))
            #kernels.append(Local_Kernel(LODE_Kernel(A_list[i], self.common_terms), num_tasks, center_list[i], weight_lengthscale, output_distance))
            kernels.append(LODE_Kernel(A_list[i], self.common_terms))
            #_kernels.append(Local_Kernel(LODE_Kernel(A_list[i], self.common_terms), num_tasks, center_list[i], weight_lengthscale, output_distance))
            means.append(Equilibrium_Mean(equilibrium_list[i], num_tasks))
            w_functions.append(Gaussian_Weight(center_list[i], weight_lengthscale))

        #self.mean_module = Global_Mean(means, num_tasks,output_distance)
        self.mean_module = Global_Mean_2(means, w_functions, num_tasks,output_distance)
        # self.covar_module = Global_Kernel(_kernels, num_tasks,output_distance)
        self.covar_module = Global_Kernel_2(kernels, w_functions, num_tasks,output_distance)

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()

        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, out=mean_x, common_terms=self.common_terms)

        # if X % 10 == 0:
        # if DEBUG:
        #     plot_weights(X, mean_x[:,0], 'Global Mean')
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
import matplotlib.pyplot as plt
def plot_weights(x, weights, title="Weighting Function"):
    plt.figure(figsize=(12, 6))
    if isinstance(weights, list):
        for i, weight in enumerate(weights):
            plt.plot(x, weight, label=f'Weight {i}')
        plt.legend()
    else:
        plt.plot(x, weights)
    plt.xlabel("x")
    plt.ylabel("Weight")
    plt.title(title)
    plt.show()

class Local_Mean(Equilibrium_Mean):
    def __init__(self, mean_values:torch.Tensor, num_tasks:int, center:torch.Tensor, weight_lengthscale:torch.Tensor, output_distance=False):
        super(Local_Mean, self).__init__(mean_values, num_tasks)
        self.weighting_function = Gaussian_Weight(center, weight_lengthscale)
        self.output_distance = output_distance

    def forward(self, x):
        if self.output_distance is False:
            return self.weighting_function(x) * super().forward(x)
        else:
            output = super().forward(x)
            return self.weighting_function(output) * output

class Global_Mean(Mean):
    def __init__(self, local_means:List[Local_Mean], num_tasks:int, output_distance:bool=False):
        super(Global_Mean, self).__init__()
        self.num_tasks = num_tasks
        self.local_means = ModuleList(local_means)
        self.output_distance = output_distance


    def forward(self, x:torch.Tensor):

        mean = sum(local_mean(x) for local_mean in self.local_means)
        distance_measure =  mean.clone() if self.output_distance else x.clone()  # FIXME: this is wrong

        weight = sum(local_mean.weighting_function(distance_measure) for local_mean in self.local_means)

        # mean = torch.zeros_like(x)
        # if DEBUG:
        #     local_means = []
        #     weights = []
        #     for local_mean in self.local_means:
        #         local_means.append(local_mean(x))
                
        #     distance_measure =  mean.clone() if self.output_distance else x.clone()
        #     for local_mean in self.local_means:
        #         weights.append(local_mean.weighting_function(distance_measure))
                
                
        #     plot_weights(x, weights)
        #     plot_weights(x, [local_means[0][:,0],local_means[1][:,0]], 'Local Means')
        #     plot_weights(x, weight, 'Sum of Weights')

        return mean / weight
    
class Global_Mean_2(Mean):
    def __init__(self, local_means:List[Mean], weight_functions:List[Gaussian_Weight], num_tasks:int, output_distance:bool=False):
        super(Global_Mean_2, self).__init__()
        self.num_tasks = num_tasks
        self.local_means = ModuleList(local_means)
        self.weight_functions = ModuleList(weight_functions)
        self.output_distance = output_distance

    def forward(self, x:torch.Tensor):
        mean_list = [local_mean(x) for local_mean in self.local_means]
        
        distance_measure =  mean_list if self.output_distance else [x.clone() for _ in range(len(mean_list))] 

        weight_list = [self.weight_functions[i](distance_measure[i]) for i in range(len(self.weight_functions))]

        mean = sum(mean_list[i] * weight_list[i] for i in range(len(mean_list)))/sum(weight_list)

        if DEBUG:
            plot_weights(x, weight_list)
            plot_weights(x, [mean_list[0][:,0],mean_list[1][:,0]], 'Local Means')
            plot_weights(x, sum(weight_list), 'Sum of Weights')

        return mean

    
class Local_Kernel(Kernel):
    def __init__(self, kernel:Kernel, num_tasks:int, center:torch.Tensor, weight_lengthscale:torch.Tensor, output_distance=False):
        super(Local_Kernel, self).__init__(active_dims=None)
        self.kernel = kernel
        self.weighting_function = Gaussian_Weight(center, weight_lengthscale)
        self.num_tasks = num_tasks
        self.output_distance = output_distance
    
    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

    def forward(self, x1, x2, diag=False, **params):
        if self.output_distance is False:
            weight_matrix_1 = torch.tile(self.weighting_function(x1),(self.num_tasks,1))
            weight_matrix_2 = torch.tile(self.weighting_function(x2),(self.num_tasks,1))
        else:
            out = params["out"]
            x1_eq_x2 = torch.equal(x1, x2)
            if (x1_eq_x2):
                out_1 = out_2 = out
            else:
                length_1 = x1.size(0)
                length_2 = x2.size(0)
                out_1 = out[:length_1]
                out_2 = out[-length_2:]

            weight_matrix_1 = torch.tile(self.weighting_function(out_1),(self.num_tasks,1))
            weight_matrix_2 = torch.tile(self.weighting_function(out_2),(self.num_tasks,1))

        return weight_matrix_1 * self.kernel(x1,x2, diag, **params) * weight_matrix_2.t()   
        
class Global_Kernel(Kernel):
    def __init__(self, local_kernels:List[Local_Kernel], num_tasks, output_distance=False):
        super(Global_Kernel, self).__init__(active_dims=None)
        self.num_tasks = num_tasks
        self.local_kernels = ModuleList(local_kernels)
        self.output_distance = output_distance

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

    def forward(self, x1, x2, diag=False, **params):
        if self.output_distance is False:
            weight = sum((local_kernel.weighting_function(x1) * local_kernel.weighting_function(x2).t())  for local_kernel in self.local_kernels)
            covar = sum(local_kernel(x1, x2, diag=False, **params) for local_kernel in self.local_kernels)
        else:
            out = params["out"]
            x1_eq_x2 = torch.equal(x1, x2)
            if (x1_eq_x2):
                out_1 = out_2 = out
            else:
                length_1 = x1.size(0)
                length_2 = x2.size(0)
                out_1 = out[:length_1]
                out_2 = out[-length_2:]

                if DEBUG:
                    
                    weights_1 = []    
                    weights_2 = []    
                    #
                    #  distance_measure =  mean.clone() if self.output_distance else x1.clone()
                    for local_kern in self.local_kernels:
                        weights_1.append(local_kern.weighting_function(out_1))
                        weights_2.append(local_kern.weighting_function(out_2))
                        
                    plot_weights(x1, weights_1)
                    plot_weights(x2, weights_2)
                    #plot_weights(x1, weight, 'Sum of Weights')


            weight = sum((local_kernel.weighting_function(out_1) * local_kernel.weighting_function(out_2).t())  for local_kernel in self.local_kernels)
            covar = sum(local_kernel(x1, x2, diag=False, **params) for local_kernel in self.local_kernels) #, out_1=out_1, out_2=out_2,

        weight_matrix = torch.tile(weight,(self.num_tasks,self.num_tasks))

        return covar / weight_matrix

        
class Global_Kernel_2(Kernel):
    def __init__(self, local_kernels:List[Kernel], weight_functions:List[Gaussian_Weight], num_tasks, output_distance=False):
        super(Global_Kernel_2, self).__init__(active_dims=None)
        self.num_tasks = num_tasks
        self.local_kernels = ModuleList(local_kernels)
        self.weight_functions = ModuleList(weight_functions)
        self.output_distance = output_distance

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

        weight_matrices_1 = [torch.tile(weighting_function(distance_measure_1),(self.num_tasks,1)) for weighting_function in self.weight_functions]
        weight_matrices_2 = [torch.tile(weighting_function(distance_measure_2),(self.num_tasks,1)) for weighting_function in self.weight_functions]

        covariances = [local_kernel(x1, x2, diag=False, **params) for local_kernel in self.local_kernels]


        weight = sum((weighting_function(distance_measure_1) * weighting_function(distance_measure_2).t())  for weighting_function in self.weight_functions)
        weight_matrix = torch.tile(weight,(self.num_tasks,self.num_tasks))
        covar = sum(weight_matrices_1[i] * covariances[i] * weight_matrices_2[i].t()  for i in range(len(covariances))) / weight_matrix

        return covar