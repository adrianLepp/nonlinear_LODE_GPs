import gpytorch 
from sage.all import *
from sage.calculus.var import var
from nonlinear_LODE_GPs.kernels import LODE_Kernel, _Diagonal_Canonical_Kernel
import pprint
import torch
from nonlinear_LODE_GPs.masking import masking, create_mask
from nonlinear_LODE_GPs.mean_modules import *
from nonlinear_LODE_GPs.likelihoods import MultitaskGaussianLikelihoodWithMissingObs

def optimize_gp(gp, training_iterations=100, verbose=True, hyperparameters:dict=None):
    # Find optimal model hyperparameters
    gp.train()
    gp.likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        # params=list(set(gp.parameters()) - {gp.pre_model.parameters() }),
        gp.parameters(), 
        lr=0.1
    )  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(gp.likelihood, gp)

    if hyperparameters is not None:
        for key, value in hyperparameters.items():
            if hasattr(gp.covar_module.model_parameters, key):
                setattr(gp.covar_module.model_parameters, key, torch.nn.Parameter(torch.tensor(value), requires_grad=False))
            else:
                print(f'Hyperparameter {key} not found in model')

    training_loss = []
    
    #print(list(self.named_parameters()))
    for i in range(training_iterations):
        
        optimizer.zero_grad()
        output = gp(gp.train_inputs[0])#FIXME: 
        loss = -mll(output, gp.train_targets)
        loss.backward()
        if verbose is True:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

        training_loss.append(loss.item())

    
    if verbose:
        print("\n----------------------------------------------------------------------------------\n")
        print('Trained model parameters:')
    named_parameters = list(gp.named_parameters())
    param_conversion = torch.nn.Softplus()
    parameters = {}

    for j in range(len(named_parameters)):
        if verbose:
            print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()
        parameters[named_parameters[j][0]] = param_conversion(named_parameters[j][1].data).tolist()
        # print(named_parameters[j][0], (named_parameters[j][1].data)) #.item()
    if verbose:
        print("\n----------------------------------------------------------------------------------\n")

    #print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        #gp.likelihood.noise = torch.tensor(1e-8)
        #gp.covar_module.model_parameters.signal_variance_3 = torch.nn.Parameter(abs(gp.covar_module.model_parameters.signal_variance_3))

    #print(list(self.named_parameters()))

    return training_loss, parameters


class LODEGP(gpytorch.models.ExactGP):
    num_tasks:int
    contains_nan:bool

    def __init__(
            self, 
            train_x:torch.Tensor, 
            train_y:torch.Tensor, 
            likelihood:gpytorch.likelihoods.Likelihood, 
            num_tasks:int, 
            A, 
            mean_module:gpytorch.means.Mean=None,
            additive_se=False,
            verbose=False,
        ):
        self.contains_nan = train_y is not None and any(train_y.isnan().flatten())
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
        
        if train_x is not None:
            self.common_terms = {
                "t_diff" : train_x-train_x.t(),
                "t_sum" : train_x+train_x.t(),
                "t_ones": torch.ones_like(train_x-train_x.t()),
                "t_zeroes": torch.zeros_like(train_x-train_x.t())
            }
        else:
            self.common_terms = {
                "t_diff" : None,
                "t_sum" : None,
                "t_ones": None,
                "t_zeroes": None
            }

        if additive_se:
            self.covar_module =gpytorch.kernels.ScaleKernel(gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=0
            )) + gpytorch.kernels.ScaleKernel(LODE_Kernel(A, self.common_terms))
        else:
            self.covar_module = LODE_Kernel(A, self.common_terms, verbose=verbose)
            #FIXME do we want rand
            for name, param in self.covar_module.named_parameters():
                param.data = torch.rand_like(param) * 3 -1.5 

    def forward(self, X, **kwargs):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X, common_terms=self.common_terms)


        if self.contains_nan and isinstance(self.likelihood, MultitaskGaussianLikelihoodWithMissingObs):   
            mean_x, covar_x = masking(base_mask=self.mask, mean=mean_x, covar=covar_x, fill_zeros=True)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)   
    
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)   
    
    def set_train_data(self, inputs=None, targets=None, strict=True):
        self.contains_nan = targets is not None and any(targets.isnan().flatten())
        if self.contains_nan and isinstance(self.likelihood, MultitaskGaussianLikelihoodWithMissingObs):
            targets, self.mask = create_mask(targets)
        self.common_terms = {
                "t_diff" : inputs-inputs.t(),
                "t_sum" : inputs+inputs.t(),
                "t_ones": torch.ones_like(inputs-inputs.t()),
                "t_zeroes": torch.zeros_like(inputs-inputs.t())
            }
        return super().set_train_data(inputs, targets, strict)

    
class Diagonal_Canonical_GP(gpytorch.models.ExactGP):
    num_tasks:int
    contains_nan:bool

    def __init__(
            self, 
            train_x:torch.Tensor, 
            train_y:torch.Tensor, 
            likelihood:gpytorch.likelihoods.Likelihood, 
            num_tasks:int, 
            eigenvec:torch.Tensor,
            eigenval:torch.Tensor,
            control:torch.Tensor,
            u, 
            mean_module:gpytorch.means.Mean=None,
            additive_se=False,
        ):
        
        self.num_tasks = num_tasks

        super(Diagonal_Canonical_GP, self).__init__(train_x, train_y, likelihood)

        if mean_module is None:
            self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ZeroMean(), num_tasks=num_tasks)
        else:
            self.mean_module = mean_module

        if additive_se:
            self.covar_module =gpytorch.kernels.ScaleKernel(gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.RBFKernel(), num_tasks=num_tasks, rank=0
            )) + gpytorch.kernels.ScaleKernel(_Diagonal_Canonical_Kernel(num_tasks, eigenvalues=eigenval, eigenvectors=eigenvec, control=control, u=u))
        else:
            self.covar_module = _Diagonal_Canonical_Kernel(num_tasks, eigenvalues=eigenval, eigenvectors=eigenvec, control=control, u=u)

    def forward(self, X):
        mean_x = self.mean_module(X)
        covar_x = self.covar_module(X)
    
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)   


    

