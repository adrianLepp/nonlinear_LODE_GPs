import gpytorch 
# from sage.all import *
from kernels import *
import torch
from mean_modules import *
from weighting import *
from lodegp import *

import matplotlib.pyplot as plt 
import seaborn as sns

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )

class Weighted_GP(LODEGP):#gpytorch.models.deep_gps.DeepGP
    def __init__(self, train_x, train_y, likelihood, num_tasks, system_matrix, equilibrium, weight_lengthscale):
        mean_module = Equilibrium_Mean(equilibrium, num_tasks)
        super().__init__(train_x, train_y, likelihood, num_tasks, system_matrix, mean_module)

        w_function = Gaussian_Weight(equilibrium, weight_lengthscale)

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
            w_functions.append(Gaussian_Weight(centers[i], weight_lengthscale))

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
    


# class Local_GP_Sum(gpytorch.Module):
class Local_GP_Sum(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_tasks, system_matrices, equilibriums, centers, weight_lengthscale=None):
        super(Local_GP_Sum, self).__init__(train_x, train_y, likelihood)
        # super(LODEGP, self).__init__(train_x, train_y, likelihood)
        
        # train_inputs = (train_x,)
        # self.train_inputs = tuple(tri.unsqueeze(-1) if tri.ndimension() == 1 else tri for tri in train_inputs)
        # self.train_targets = train_y

        self.num_tasks = num_tasks

        models = []
        w_fcts = []

        for i in range(len(system_matrices)):
            mean_module = Equilibrium_Mean(equilibriums[i], num_tasks)
            model = LODEGP(train_x, train_y, likelihood, num_tasks, system_matrices[i], mean_module)

            w_fcts.append(Gaussian_Weight(centers[i]))#, weight_lengthscale
            # w_fcts[i].initialize(length=torch.tensor(weight_lengthscale, requires_grad=True))#TODO
            w_fcts[i].length = weight_lengthscale
            models.append(model)
        
        model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, num_tasks)
        models.append(model)
        w_fct = Constant_Weight()
        w_fct.initialize(weight=torch.tensor(0.5, requires_grad=True))
        w_fcts.append(w_fct)

        self.models = ModuleList(models)
        self.w_fcts = ModuleList(w_fcts)

        # self.likelihood =likelihood# gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks, noise_constraint=gpytorch.constraints.Positive())
    
    def _optimize(self, model, training_iterations=100, verbose=False):
        for model in self.models:
            optimize_gp(model, training_iterations=training_iterations, verbose=verbose)

            # print("\n----------------------------------------------------------------------------------\n")
            # print(f'Trained model parameters:')
            # named_parameters = list(model.named_parameters())
            # param_conversion = torch.nn.Softplus()

            # for j in range(len(named_parameters)):
            #     print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()
            # print("\n----------------------------------------------------------------------------------\n")

    def optimize(self, training_iterations=100, verbose=False):
        self.train()
        self.likelihood.train()

        # Use the adam optimizer
        # bias_params = [p for name, p in self.named_parameters() if 'bias' in name]
        # optimizer = torch.optim.Adam(
        #     [
        #         {'params': self.models[0].parameters()},
        #         {'params': self.w_fcts[0].parameters()},
        #         {'params': self.models[1].parameters()},
        #         {'params': self.w_fcts[1].parameters()},
        #         {'params': self.models[2].parameters()},
        #         {'params': self.w_fcts[2].parameters()}
        #     ] ,
        #     # params= (self.models[i].parameters(), self.w_fcts[i].parameters() for i in range(len(self.models))),
        #     # self.parameters()
        #     lr=0.1
        # )

        # optimizer = torch.optim.Adam([{'params':self.parameters()}, {'params':self.w_fcts[0].parameters()}], lr=0.1)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)

        # "Loss" for GPs - the marginal log likelihood
        #mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        training_loss = []
        
        #print(list(self.named_parameters()))
        for i in range(training_iterations):
            
            optimizer.zero_grad()
            output = self(self.train_inputs[0])
            loss = -mll(output, self.train_targets)
            loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

            training_loss.append(loss.item())
            # for i in range(len(self.w_fcts)-1):
            #     print(f'Grad {self.w_fcts[i].raw_length.grad.view(-1).item()}')

        print("\n----------------------------------------------------------------------------------\n")
        print('Trained model parameters:')
        named_parameters = list(self.named_parameters())
        param_conversion = torch.nn.Softplus()

        for j in range(len(named_parameters)):
            print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()
            # print(named_parameters[j][0], (named_parameters[j][1].data)) #.item()
        print("\n----------------------------------------------------------------------------------\n")

    def eval(self):
        for model in self.models:
            model.eval()
            model.likelihood.eval()
        self.likelihood.eval()

    def train(self):
        for model in self.models:
            model.train()
            model.likelihood.train()

    def set_train_data(self, x, y, **kwargs):
        [model.set_train_data(x, y, **kwargs) for model in self.models]
        

    def predict(self, x, noise:torch.Tensor=None):
        # with torch.no_grad():
        outputs = [self.likelihood(self.models[l](x), noise=noise) for l in range(len(self.models))]
        weights = [self.w_fcts[l](output.mean) for l, output in enumerate(outputs)]
        # weights_extended = [torch.tile(weights[l],(self.num_tasks,1)) for l in range(len(weights))]
        weights_extended = [weights[l].repeat_interleave(self.num_tasks) for l in range(len(weights))] 
        mean = sum([outputs[l].mean * weights[l] for l in range(len(outputs))]) /sum(weights)
        cov = sum([outputs[l].covariance_matrix * weights_extended[l] for l in range(len(outputs))])

            # hm = sns.heatmap(cov,
            #      cbar=True,
            #     #  annot=True,
            #      square=True,
            #     #  fmt='.2f',
            #     #  annot_kws={'size': 12},
            #     #  yticklabels=cols,
            #     #  xticklabels=cols
            #     )
                
            # plt.tight_layout()
            # plt.show()
        return mean, cov, weights
    
    def forward(self, x, noise:torch.Tensor=None):
        mean, cov, weights = self.predict(x, noise=noise)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, cov)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, torch.eye(mean.shape[0]*mean.shape[1]))
