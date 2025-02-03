import gpytorch 
# from sage.all import *
from kernels import *
import torch
from mean_modules import *
from weighting import *
from lodegp import *

class Weighted_GP(LODEGP):#gpytorch.models.deep_gps.DeepGP
    def __init__(self, train_x, train_y, likelihood, num_tasks, system_matrix, equilibrium, weight_lengthscale):
        mean_module = Equilibrium_Mean(equilibrium, num_tasks)
        super().__init__(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)

        w_function = Weighting_Function(equilibrium, weight_lengthscale)

        #super().__init__(train_inputs=train_x, train_targets=train_y, likelihood=likelihood)
        
        self.w_fct = w_function

        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, X):
        if not torch.equal(X, self.train_inputs[0]):
            self.common_terms["t_diff"] = X-X.t()
            self.common_terms["t_sum"] = X+X.t()
        mean_x = self.mean_module(X)
        
        covar_x = self.covar_module(X, common_terms=self.common_terms)
        

        if self.training:
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
        else:
            weight = self.w_fct(mean_x)
            mean_weighted = weight * mean_x
            self.weight = weight
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_weighted, covar_x)

    def _forward(self, inputs):
        lode_out = self.lode_gp(inputs)
        # output = self.w_fct(lode_out)
        # return output

        weighted_mean = self.w_fct(lode_out.mean) * lode_out.mean 
        
            #or 
        return gpytorch.distributions.MultitaskMultivariateNormal(weighted_mean, lode_out.covariance_matrix)   
    
class Weighted_Sum_GP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, system_matrices, equilibriums, centers, weight_lengthscale):
        super(Weighted_Sum_GP, self).__init__(train_x, train_y, likelihood)

        #https://docs.gpytorch.ai/en/v1.12/examples/05_Deep_Gaussian_Processes/DGP_Multitask_Regression.html
        #https://docs.gpytorch.ai/en/latest/examples/05_Deep_Gaussian_Processes/Deep_Gaussian_Processes.html
        models = []
        weighted_models = []
        w_functions = []
        for i in range(len(system_matrices)):
            mean_module = Equilibrium_Mean(equilibriums[i], num_tasks)
            model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrices[i], mean_module)
            weighted_model = Weighted_GP(train_x, train_y, likelihood, num_tasks, system_matrices[i], equilibriums[i], weight_lengthscale)
            optimize_gp(weighted_model, training_iterations=100, verbose=False)
            w_functions.append(Weighting_Function(centers[i], weight_lengthscale))

            weighted_models.append(weighted_model)
            models.append(model)


        self.models = ModuleList(models)
        self.w_functions = ModuleList(w_functions)
        self.weighted_models = ModuleList(weighted_models)
        
    def eval(self, **kwargs):
        #super().eval(**kwargs)
        for model in self.models:
            model.eval()
        for model in self.weighted_models:
            model.eval()

    def train(self,args, kwargs):
        #super().train(*args, **kwargs)
        for model in self.models:
            model.train(*args, **kwargs)
        for model in self.weighted_models:
            model.train(*args, **kwargs)
        

    def forward(self, x):
        outputs = [self.models[i](x) for i in range(len(self.models))]
        weights = [self.w_functions[i](self.likelihood(output).mean) for i, output in enumerate(outputs)]

        weighted_outputs = [self.weighted_models[i](x) for i in range(len(self.models))]
        weights = [self.weighted_models[i].weight for i in range(len(self.models))]
        
        #out = torch.sum(outputs.mean * weights)/torch.sum(weights)
        out = sum(weighted_outputs)
        weight = sum(weights)
        clean_mean = out.mean / weight

        return gpytorch.distributions.MultitaskMultivariateNormal(clean_mean, out.covariance_matrix)
        #TODO: divide mean by sum of weights
        return out
    


class Local_GP_Sum(gpytorch.Module):
    def __init__(self, train_x, train_y, likelihood, num_tasks, system_matrices, equilibriums, centers, weight_lengthscale=None):
        super(Local_GP_Sum, self).__init__()

        self.num_tasks = num_tasks

        models = []
        w_fcts = []

        for i in range(len(system_matrices)):
            mean_module = Equilibrium_Mean(equilibriums[i], num_tasks)
            model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrices[i], mean_module)

            w_fcts.append(Weighting_Function(centers[i]))#, weight_lengthscale
            w_fcts[i].initialize(length=torch.tensor(weight_lengthscale, requires_grad=False))
            #w_fcts[i].lengthscale = weight_lengthscale
            models.append(model)

        self.models = ModuleList(models)
        self.w_fcts = ModuleList(w_fcts)

        self.likelihood =likelihood# gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
    
    def optimize(self, model, training_iterations=100, verbose=False):
        for model in self.models:
            optimize_gp(model, training_iterations=training_iterations, verbose=verbose)

            print("\n----------------------------------------------------------------------------------\n")
            print(f'Trained model parameters:')
            named_parameters = list(model.named_parameters())
            param_conversion = torch.nn.Softplus()

            for j in range(len(named_parameters)):
                print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()
            print("\n----------------------------------------------------------------------------------\n")

    def eval(self):
        for model in self.models:
            model.eval()
            model.likelihood.eval()
        self.likelihood.eval()

    def set_train_data(self, x, y, **kwargs):
        [model.set_train_data(x, y, **kwargs) for model in self.models]
        

    def predict(self, x, noise:torch.Tensor=None):
        with torch.no_grad():
            outputs = [self.likelihood(self.models[l](x), noise=noise) for l in range(len(self.models))]
            weights = [self.w_fcts[l](output.mean) for l, output in enumerate(outputs)]
            out = sum([outputs[l].mean * weights[l] for l in range(len(outputs))])/sum(weights)
        return out, weights
    
    def forward(self, x, noise:torch.Tensor=None):
        out, weights = self.predict(x, noise=noise)
        return gpytorch.distributions.MultitaskMultivariateNormal(out, torch.eye(out.shape[0]*out.shape[1]))
