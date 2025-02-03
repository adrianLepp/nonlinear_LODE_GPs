import gpytorch 
from sage.all import *
from sage.calculus.var import var
from kernels import *
import pprint
import torch
from masking import masking, create_mask
from mean_modules import *
from likelihoods import MultitaskGaussianLikelihoodWithMissingObs

def optimize_gp(gp, training_iterations=100, verbose=True, hyperparameters:dict=None):
    # Find optimal model hyperparameters
    gp.train()
    gp.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    if hyperparameters is not None:
        for key, value in hyperparameters.items():
            if hasattr(gp.covar_module.model_parameters, key):
                setattr(gp.covar_module.model_parameters, key, torch.nn.Parameter(torch.tensor(value), requires_grad=False))
            else:
                print(f'Hyperparameter {key} not found in model')
    
    #print(list(self.named_parameters()))
    for i in range(training_iterations):
        
        optimizer.zero_grad()
        output = gp(gp.train_inputs[0])#FIXME: 
        loss = -mll(output, gp.train_targets)
        loss.backward()
        if verbose is True:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    

    print("\n----------------------------------------------------------------------------------\n")
    print('Trained model parameters:')
    named_parameters = list(gp.named_parameters())
    #param_conversion = torch.nn.Softplus()

    for j in range(len(named_parameters)):
        #print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()
        print(named_parameters[j][0], (named_parameters[j][1].data)) #.item()
    print("\n----------------------------------------------------------------------------------\n")

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
    contains_nan:bool

    def __init__(self, train_x:torch.Tensor, train_y:torch.Tensor, likelihood:gpytorch.likelihoods.Likelihood, num_tasks:int, A, mean_module:gpytorch.means.Mean=None):
        self.contains_nan = any(train_y.isnan().flatten())
        self.num_tasks = num_tasks

        if self.contains_nan and isinstance(likelihood, MultitaskGaussianLikelihoodWithMissingObs):
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


        if self.contains_nan and isinstance(self.likelihood, MultitaskGaussianLikelihoodWithMissingObs):   
            mean_x, covar_x = masking(base_mask=self.mask, mean=mean_x, covar=covar_x, fill_zeros=True)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)   
    
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
    

