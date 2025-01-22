import gpytorch 
from gpytorch.kernels import MultitaskKernel, Kernel, RBFKernel
from gpytorch.means import Mean
from gpytorch.kernels.kernel import sq_dist, dist
from gpytorch.functions import RBFCovariance
from sage.all import *
import sage
from sage.calculus.var import var
from kernels import *
import pprint
import time
import torch
from masking import masking, create_mask
from likelihoods import MultitaskGaussianLikelihoodWithMissingObs
from noise_models import MaskedNoise
from mean_modules import *

def optimize_gp(gp, training_iterations=100, verbose=True):
    # Find optimal model hyperparameters
    gp.train()
    gp.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)
    #print(list(self.named_parameters()))
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = gp(gp.train_inputs[0])#FIXME: 
        loss = -mll(output, gp.train_targets)
        loss.backward()
        if verbose is True:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        #gp.likelihood.noise = torch.tensor(1e-8)
        #gp.covar_module.model_parameters.signal_variance_3 = torch.nn.Parameter(abs(gp.covar_module.model_parameters.signal_variance_3))

    #print(list(self.named_parameters()))

class LODEGP_Deprecated(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, A):
        super(LODEGP_Deprecated, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        self.model_parameters = torch.nn.ParameterDict()
        #self.num_tasks = num_tasks

        D, U, V = A.smith_form()
        print(f"D:{D}")
        print(f"V:{V}")
        x, a, b = var("x, a, b")
        V_temp = [list(b) for b in V.rows()]
        #print(V_temp)
        V = sage_eval(f"matrix({str(V_temp)})", locals={"x":x, "a":a, "b":b})
        Vt = V.transpose()
        kernel_matrix, self.kernel_translation_dict, parameter_dict = create_kernel_matrix_from_diagonal(D)
        self.ode_count = A.nrows()
        self.kernelsize = len(kernel_matrix)
        self.model_parameters.update(parameter_dict)
        #print(self.model_parameters)
        var(["x", "dx1", "dx2"] + ["t1", "t2"] + [f"LODEGP_kernel_{i}" for i in range(len(kernel_matrix[Integer(0)]))])
        k = matrix(Integer(len(kernel_matrix)), Integer(len(kernel_matrix)), kernel_matrix)
        V = V.substitute(x=dx1)
        Vt = Vt.substitute(x=dx2)

        #train_x = self._slice_input(train_x)

        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }
        self.V = V
        self.matrix_multiplication = matrix(k.base_ring(), len(k[0]), len(k[0]), (V*k*Vt))
        self.diffed_kernel = differentiate_kernel_matrix(k, V, Vt, self.kernel_translation_dict)
        self.sum_diff_replaced = replace_sum_and_diff(self.diffed_kernel)
        self.covar_description = translate_kernel_matrix_to_gpytorch_kernel(self.sum_diff_replaced, self.model_parameters, common_terms=self.common_terms)
        self.covar_module = LODE_Kernel(self.covar_description, self.model_parameters)


    def __str__(self, substituted=False):
        if substituted:
            return pprint.pformat(str(self.sum_diff_replaced), indent=self.kernelsize)
        else:
            return pprint.pformat(str(self.diffed_kernel), indent=self.kernelsize)

    def __latexify_kernel__(self, substituted=False):
        if substituted:
            return pprint.pformat(latex(self.sum_diff_replaced), indent=self.kernelsize)
        else:
            return pprint.pformat(latex(self.diffed_kernel), indent=self.kernelsize)

    def __pretty_print_kernel__(self, substituted=False):
        if substituted:
            return pprint.pformat(pretty_print(self.matrix_multiplication), indent=self.kernelsize)
        else:
            pretty_print(self.matrix_multiplication)
            print(str(self.kernel_translation_dict))

    def _slice_input(self, X):
            r"""
            Slices :math:`X` according to ``self.active_dims``. If ``X`` is 1D then returns
            a 2D tensor with shape :math:`N \times 1`.
            :param torch.Tensor X: A 1D or 2D input tensor.
            :returns: a 2D slice of :math:`X`
            :rtype: torch.Tensor
            """
            if X.dim() == 2:
                #return X[:, self.active_dims]
                return X[:, 0]
            elif X.dim() == 1:
                return X.unsqueeze(1)
            else:
                raise ValueError("Input X must be either 1 or 2 dimensional.")

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)
        #print(torch.linalg.eigvalsh(covar_x.evaluate()))
        #covar_x = covar_x.flatten()
        #print(list(torch.linalg.eigh(covar_x)[0])[::-1])
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 


class LODEGP(gpytorch.models.ExactGP):
    num_tasks:int
    contains_nan:bool#MultitaskGaussianLikelihoodWithMissingObs

    def __init__(self, train_x:torch.Tensor, train_y:torch.Tensor, likelihood:gpytorch.likelihoods.Likelihood, num_tasks:int, A, mean_module:gpytorch.means.Mean=None):
        self.contains_nan = any(train_y.isnan().flatten())#MultitaskGaussianLikelihoodWithMissingObs
        self.num_tasks = num_tasks

        #MultitaskGaussianLikelihoodWithMissingObs
        if self.contains_nan:
            train_y, self.mask = create_mask(train_y)

        super(LODEGP, self).__init__(train_x, train_y, likelihood)

        if mean_module is None:
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ZeroMean(), num_tasks=num_tasks
            )
        else:
            self.mean_module = mean_module
        
        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }
        
        self.covar_module = LODE_Kernel(A, self.common_terms)

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)

        #MultitaskGaussianLikelihoodWithMissingObs
        if self.contains_nan:   
            mean_x, covar_x = masking(base_mask=self.mask, mean=mean_x, covar=covar_x, fill_zeros=True)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)   
        else:   
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)   

    

class Param_LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, A, x0):
        super(Param_LODEGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        
        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }
        
        self.covar_module = Param_LODE_Kernel(A, x0, self.common_terms)

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 
    
class Changepoint_LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, A_list, changepoints:List[float]):
        super(Changepoint_LODEGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ZeroMean(), num_tasks=num_tasks
        )
        
        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }

        covar_modules = []
        for A in A_list:
            covar_modules.append(LODE_Kernel(A, self.common_terms))
        

        self.covar_module = Drastic_changepoint_Kernel(covar_modules, changepoints, num_tasks)

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x) 
    
class Sum_LODEGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks:int, A_list:List, equilibrium_list:List[torch.Tensor], center_list:List[float], weight_lengthscale:torch.Tensor):
        self.num_tasks = num_tasks
        super(Sum_LODEGP, self).__init__(train_x, train_y, likelihood)
        
        self.common_terms = {
            "t_diff" : train_x-train_x.t(),
            "t_sum" : train_x+train_x.t(),
            "t_ones": torch.ones_like(train_x-train_x.t()),
            "t_zeroes": torch.zeros_like(train_x-train_x.t())
        }

        #self.weighting_function = Weighting_Function(torch.Tensor([[100]]))


        #self.mean_module = Equilibrium_Mean(equilibrium, num_tasks) * gpytorch.means.MultitaskMean(gpytorch.means.ZeroMean(), num_tasks=num_tasks)
        #self.mean_module = Equilibrium_Mean(equilibrium, num_tasks) + gpytorch.means.MultitaskMean(Weigthing_Mean(self.weighting_function), num_tasks) 
        #self.mean_module = Local_Mean(equilibrium, num_tasks, torch.Tensor([[100]]))
        means = []
        kernels = []
        for i in range(len(A_list)):
            means.append(Local_Mean(equilibrium_list[i], num_tasks, torch.Tensor([[center_list[i]]]), weight_lengthscale))
            kernels.append(Local_Kernel(LODE_Kernel(A_list[i], self.common_terms), num_tasks, torch.Tensor([[center_list[i]]]), weight_lengthscale))

        # self.mean_module = Global_Mean([
        #     Local_Mean(equilibrium, num_tasks, torch.Tensor([[t_1]])),
        #     Local_Mean(equilibrium, num_tasks, torch.Tensor([[t_2]]))
        #     ], num_tasks)
        
        #self.covar_module = MultitaskKernel(RBFKernel(), num_tasks) * LODE_Kernel(A, self.common_terms)
        #self.covar_module = Local_Kernel(LODE_Kernel(A, self.common_terms), torch.Tensor([[0]]), num_tasks) 

        # self.covar_module = Global_Kernel([
        #     Local_Kernel(LODE_Kernel(A, self.common_terms), torch.Tensor([[t_1]]), num_tasks), 
        #     Local_Kernel(LODE_Kernel(A, self.common_terms), torch.Tensor([[t_2]]), num_tasks), 
        # ], num_tasks)

        self.mean_module = Global_Mean(means, num_tasks)
        self.covar_module = Global_Kernel(kernels, num_tasks)

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        # weight_matrix = self.weighting_function(X)
        # weight_matrix_ = torch.tile(weight_matrix,(self.num_tasks,1))
        # mean_x = weight_matrix * self.mean_module(X)
        # covar_x = weight_matrix_.t() * self.covar_module(X, common_terms=self.common_terms) * weight_matrix_
        # dist = gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        # for i in range(dist.mean.shape[0]):
        #     dist[i] = dist[i] / weight_matrix_[i,0].detach().item()

        # return dist
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    

class Weighting_Function(gpytorch.Module):#gpytorch.Module
    def __init__(self, center:torch.Tensor, lengthscale:torch.Tensor):
        super(Weighting_Function, self).__init__()
        self.center = center
        #self.lengthscale = torch.nn.Parameter(torch.ones(1)*(44194))
        self.lengthscale = lengthscale


    def forward(self, x):
        center = self.center
        
        x_ = x.div(self.lengthscale)
        center_ = center.div(self.lengthscale)
        unitless_sq_dist = self.covar_dist(x_, center_)
        # clone because inplace operations will mess with what's saved for backward
        covar_mat = unitless_sq_dist.div_(-2.0).exp_()
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
    
class Local_Mean(Equilibrium_Mean):
    def __init__(self, mean_values:torch.Tensor, num_tasks:int, center:torch.Tensor, weight_lengthscale:torch.Tensor):
        super(Local_Mean, self).__init__(mean_values, num_tasks)
        self.weighting_function = Weighting_Function(center, weight_lengthscale)

    def forward(self, x):
        return self.weighting_function(x) * super().forward(x)


class Local_Kernel(Kernel):
    def __init__(self, kernel:Kernel, num_tasks:int, center:torch.Tensor, weight_lengthscale:torch.Tensor):
        super(Local_Kernel, self).__init__(active_dims=None)
        self.kernel = kernel
        self.weighting_function = Weighting_Function(center, weight_lengthscale)
        self.num_tasks = num_tasks
    
    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

    def forward(self, x1, x2, diag=False, **params):
        weight_matrix_1 = torch.tile(self.weighting_function(x1),(self.num_tasks,1))
        weight_matrix_2 = torch.tile(self.weighting_function(x2),(self.num_tasks,1))

        return weight_matrix_1 * self.kernel(x1,x2, diag, **params) * weight_matrix_2.t()

class Global_Mean(Mean):
    def __init__(self, local_means:List[Local_Mean], num_tasks:int):
        super(Global_Mean, self).__init__()
        self.num_tasks = num_tasks
        self.local_means = ModuleList(local_means)


    def forward(self, x):
        weight = sum(local_mean.weighting_function(x) for local_mean in self.local_means)
        mean = sum(local_mean(x) for local_mean in self.local_means)

        # mean = torch.zeros_like(x)
        # for local_mean in self.local_means:
        #     weight += local_mean.weighting_function(x)
        #     mean += local_mean(x)
        return mean / weight
    
    
class Global_Kernel(Kernel):
    def __init__(self, local_kernels:List[Local_Kernel], num_tasks):
        super(Global_Kernel, self).__init__(active_dims=None)
        self.num_tasks = num_tasks
        self.local_kernels = ModuleList(local_kernels)

    def num_outputs_per_input(self, x1, x2):
        """
        Given `n` data points `x1` and `m` datapoints `x2`, this multitask
        kernel returns an `(n*num_tasks) x (m*num_tasks)` covariance matrix.
        """
        return self.num_tasks

    def forward(self, x1, x2, diag=False, **params):
        weight = sum((local_kernel.weighting_function(x1) * local_kernel.weighting_function(x2).t())  for local_kernel in self.local_kernels)
        #weight_vector = torch.tile(weight,(self.num_tasks,1))
        weight_matrix = torch.tile(weight,(self.num_tasks,self.num_tasks))
        covar = sum(local_kernel(x1, x2, diag=False, **params) for local_kernel in self.local_kernels)

        return covar / weight_matrix
        