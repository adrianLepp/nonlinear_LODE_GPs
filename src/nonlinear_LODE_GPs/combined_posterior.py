import gpytorch 
from linear_operator.operators import DiagLinearOperator
# from sage.all import *
from nonlinear_LODE_GPs.kernels import *
import torch
from nonlinear_LODE_GPs.mean_modules import *
from nonlinear_LODE_GPs.weighting import *
from nonlinear_LODE_GPs.lodegp import *
from nonlinear_LODE_GPs.gp import BatchIndependentMultitaskGPModel

class CombinedPosterior_ELODEGP(gpytorch.models.ExactGP):
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
            weight_lengthscale=None,
            shared_weightscale=False,
            additive_se=False,
            clustering=False,
            output_weights=True,
            ):
        super(CombinedPosterior_ELODEGP, self).__init__(train_x, train_y, likelihood)
        # super(CombinedPosterior_ELODEGP, self).__init__()
        # self.likelihood = likelihood
        # self.train_inputs = (train_x, )
        # self.train_targets = train_y
        self.num_tasks = num_tasks

        models = []
        w_fcts = []

        self.shared_weightscale = shared_weightscale
        self.additive_se = additive_se
        self.output_weights = output_weights

        if shared_weightscale:
            self.register_parameter( name='raw_scale', parameter=torch.nn.Parameter(torch.ones(1, 1)))
            scale_constraint = gpytorch.constraints.Positive()
            self.register_constraint("raw_scale", scale_constraint)


            if weight_lengthscale is not None and not isinstance(weight_lengthscale, list):
                self.scale = torch.tensor(weight_lengthscale)#, requires_grad=True
                # self.raw_scale.requires_grad = False

        
        else: 
            self.register_parameter( name='raw_scale', parameter=torch.nn.Parameter(None))

        self.train_data_subsets = []

        self.true_centers = []

        if clustering:
            distances = torch.cdist(train_y.unsqueeze(0), torch.stack(centers).unsqueeze(0)).squeeze(0)
            cluster_assignments = torch.argmin(distances, dim=0)
            for i in range(len(system_matrices)):
                cluster_indices = (cluster_assignments == i).nonzero(as_tuple=True)[0]
                self.train_data_subsets.append((train_x[cluster_indices], train_y[cluster_indices]))
                if not self.output_weights:
                    centers[i] = train_x[cluster_indices[cluster_indices.shape[0]//2]].unsqueeze(0).unsqueeze(0)
                    self.true_centers.append(train_y[cluster_indices[cluster_indices.shape[0]//2]])
                    
                

        for i in range(len(system_matrices)):
            if clustering:
                train_x_subset, train_y_subset = self.train_data_subsets[i]
            else:
                train_x_subset, train_y_subset = train_x, train_y
            mean_module = Equilibrium_Mean(equilibriums[i], num_tasks)
            model = LODEGP(train_x_subset, train_y_subset, likelihood, num_tasks, system_matrices[i], mean_module)

            w_fcts.append(Weight_Model(centers[i], shared_weightscale=shared_weightscale, scale=weight_lengthscale[i]))
            # if weight_lengthscale is not None:
            #     w_fcts[i].initialize(scale=torch.tensor(weight_lengthscale))#TODO
            
            # w_fcts[i].length = weight_lengthscale# TODO: add weight model specific paremeter beforehand to the model
            models.append(model)
        
        if additive_se:
            model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood, num_tasks)
            models.append(model)
            w_fct = Constant_Weight()
            w_fct.initialize(weight=torch.tensor(0.5/len(system_matrices), requires_grad=False))
            # w_fct.weight.requires_grad = False
            w_fct.raw_weight.requires_grad = False
            w_fcts.append(w_fct)

        self.models = ModuleList(models)
        self.w_fcts = ModuleList(w_fcts)
    
    def _optimize(self, model, training_iterations=100, verbose=False):
        for model in self.models:
            optimize_gp(model, training_iterations=training_iterations, verbose=verbose)

    def optimize(self, training_iterations=100, verbose=False, learning_rate=0.1):
        self.train()
        self.likelihood.train()
        for model in self.models:
            model.eval()
            model.likelihood.eval()

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # optimizer = torch.optim.Adam(
        #     list(self.w_fcts[0].parameters()) +
        #     list(self.w_fcts[1].parameters()) +
        #     list(self.w_fcts[2].parameters()) +
        #     list(self.w_fcts[3].parameters()) +
        #     list(self.w_fcts[4].parameters()) 
        #     , lr=learning_rate)
            

        # mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        mll = gpytorch.mlls.LeaveOneOutPseudoLikelihood(self.likelihood, self)

        training_loss = []
        
        for i in range(training_iterations):
            
            optimizer.zero_grad()
            mean, cov, weight_list = self.predict(self.train_inputs[0])
            cov = cov.add_jitter()
            cov = cov + torch.eye(cov.size(-1)) * 1e-4
            (cov + cov.transpose(-1, -2)) / 2
            # output =  gpytorch.distributions.MultitaskMultivariateNormal(mean, cov)
            # output = gpytorch.distributions.MultitaskMultivariateNormal(mean, cov + torch.eye(cov.shape[0])* 1e-8) # + torch.eye(cov.shape[0])* 1e-8
            # loss = -mll(output, self.train_targets)
            # loss = ((mean - self.train_targets)**2).mean()
            loss = torch.cdist(mean, self.train_targets, p=2).mean()
            
            weights = torch.stack(weight_list, dim=1)
            weight_sums = weights.sum(dim=1)
            weight_loss = ((weight_sums - 1) ** 2)
            
            weight_loss_max = weight_loss.max()
            weight_loss_mean = weight_loss.mean()

            total_loss = weight_loss_mean# + loss
            total_loss.backward()

            if verbose is True:
                print('Iter %d/%d - Loss: %.3f - weight-loss %.3f' % (i + 1, training_iterations, loss.item(), weight_loss_mean.item()))
            optimizer.step()

            training_loss.append(total_loss.item())

        named_parameters = list(self.named_parameters())
        param_conversion = torch.nn.Softplus()
        parameters = {}
        for j in range(len(named_parameters)):
            parameters[named_parameters[j][0]] = param_conversion(named_parameters[j][1].data).tolist()
            # print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()
        if verbose is True:
            print("\n----------------------------------------------------------------------------------\n")
            print('Trained model parameters:')
            for j in range(len(named_parameters)):
                    print(named_parameters[j][0], param_conversion(named_parameters[j][1].data)) #.item()
                # print(named_parameters[j][0], (named_parameters[j][1].data)) #.item()
            print("\n----------------------------------------------------------------------------------\n")

        return training_loss, parameters

        

    def eval(self):
        for model in self.models:
            model.eval()
            model.likelihood.eval()
        self.likelihood.eval()
        # self.eval()

    def train(self):
        for model in self.models:
            model.train()
            model.likelihood.train()
        # self.train()
        # self.likelihood.train()

    def set_train_data(self, x, y, **kwargs):
        [model.set_train_data(x, y, **kwargs) for model in self.models]
        

    def predict(self, _x, noise:torch.Tensor=None):
        x = _x.squeeze() #TODO: is this ok?
        outputs = [self.models[l](x) for l in range(len(self.models))]
        

        if self.output_weights:
            if self.additive_se:
                weights = [self.w_fcts[l](outputs[-1], self.scale) for l, output in enumerate(outputs)]
            else:
                weights = [self.w_fcts[l](output, self.scale) for l, output in enumerate(outputs)]
        else:
            weights = [self.w_fcts[l](x.unsqueeze(1), self.scale) for l, output in enumerate(outputs)]

        weight_sum = sum(weights)
        weights_normalized = [weights[l] / weight_sum for l in range(len(weights))]

        weights_extended = [weights_normalized[l].repeat_interleave(self.num_tasks) for l in range(len(weights_normalized))] 
        mean = sum([outputs[l].mean * weights_normalized[l] for l in range(len(outputs))]) 
        mean_difference = [outputs[l].mean.flatten() - mean.flatten() for l in range(len(outputs))]
        
        cov = sum([DiagLinearOperator(weights_extended[l]) @ (outputs[l]._covar + mean_difference[l].unsqueeze(1) @ mean_difference[l].unsqueeze(0) ) for l in range(len(outputs))])

        # weights_extended = [weights[l].repeat_interleave(self.num_tasks) for l in range(len(weights))] 
        # mean = sum([outputs[l].mean * weights[l] for l in range(len(outputs))]) /sum(weights)
        # cov = sum([outputs[l].covariance_matrix @ torch.diag(weights_extended[l]) for l in range(len(outputs))]) #torch.diag(weights_extended[l]) @ 
        return mean, cov, weights
    
    def forward(self, x, noise:torch.Tensor=None):
        mean, cov, weights = self.predict(x, noise=noise)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean, cov)

    @property
    def scale(self):
        if hasattr(self, "raw_scale_constraint"):
            return self.raw_scale_constraint.transform(self.raw_scale)
        return self.raw_scale

    @scale.setter
    def scale(self, value):
        return self._set_scale(value)

    def _set_scale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_scale)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        if hasattr(self, "raw_scale_constraint"):
            self.initialize(raw_scale=self.raw_scale_constraint.inverse_transform(value))
        else:
            self.initialize(raw_scale=value)