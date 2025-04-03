import gpytorch 
# from sage.all import *
from nonlinear_LODE_GPs.kernels import *
import torch
from nonlinear_LODE_GPs.mean_modules import *
from nonlinear_LODE_GPs.weighting import *
from nonlinear_LODE_GPs.lodegp import *
from nonlinear_LODE_GPs.gp import BatchIndependentMultitaskGPModel

class Local_GP_Sum(gpytorch.models.ExactGP):
    def __init__(
            self, 
            train_x, 
            train_y, 
            likelihood, 
            num_tasks, 
            system_matrices, 
            equilibriums, 
            centers, 
            Weight_Model:Gaussian_Weight,
            weight_lengthscale=None
            ):
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

            w_fcts.append(Weight_Model(centers[i]))
            # w_fcts[i].initialize(length=torch.tensor(weight_lengthscale, requires_grad=True))#TODO
            w_fcts[i].length = weight_lengthscale# TODO: add weight model specific paremeter beforehand to the model
            models.append(model)
        
        model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, num_tasks)
        models.append(model)
        w_fct = Constant_Weight()
        w_fct.initialize(weight=torch.tensor(0.5/len(system_matrices), requires_grad=True))
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
        for model in self.models:
            model.eval()
            model.likelihood.eval()

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
            mean, cov, weights = self.predict(self.train_inputs[0])
            # output = gpytorch.distributions.MultitaskMultivariateNormal(mean, cov + torch.eye(cov.shape[0])* 1e-8)
            # loss = -mll(output, self.train_targets)
            weight_loss = torch.square(sum(weights) - 1)
            total_loss = - torch.sum(weight_loss/ weight_loss.shape[0]) # + loss
            total_loss.backward()

            # weight_loss =  sum(weights) - 1 
            # loss.backward()
            if verbose is True:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, total_loss.item()))
            optimizer.step()

            training_loss.append(total_loss.item())
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
        outputs = [self.models[l].likelihood(self.models[l](x), noise=noise) for l in range(len(self.models))]
        #weights = [self.w_fcts[l](output.mean) for l, output in enumerate(outputs)]
        weights = [self.w_fcts[l](output) for l, output in enumerate(outputs)] #TODO
        # weights_extended = [torch.tile(weights[l],(self.num_tasks,1)) for l in range(len(weights))]
        weights_extended = [weights[l].repeat_interleave(self.num_tasks) for l in range(len(weights))] 
        mean = sum([outputs[l].mean * weights[l] for l in range(len(outputs))]) /sum(weights)
        cov = sum([outputs[l].covariance_matrix * weights_extended[l] for l in range(len(outputs))])
        return mean, cov, weights
    
    def forward(self, x, noise:torch.Tensor=None):
        mean, cov, weights = self.predict(x, noise=noise)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, cov)
