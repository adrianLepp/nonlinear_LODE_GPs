from gpytorch.likelihoods.multitask_gaussian_likelihood import _MultitaskGaussianLikelihoodBase, MultitaskGaussianLikelihood
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.lazy import ConstantDiagLazyTensor, KroneckerProductLazyTensor, DiagLazyTensor
from linear_operator import LinearOperator, operators
#from gpytorch.operators import ConstantDiagLinearOperator
import torch
from noise_models import MaskedManualNoise, ManualNoise
from gpytorch.distributions import MultivariateNormal
from typing import Any
import gpytorch.settings as settings

class FixedTaskNoiseMultitaskLikelihood(_MultitaskGaussianLikelihoodBase):
    def __init__(self, noise, *args, **kwargs):
        noise_covar = FixedGaussianNoise(noise=noise)
        super().__init__(noise_covar=noise_covar, *args, **kwargs)
        self.has_global_noise = False
        self.has_task_noise = False
        
    def _shaped_noise_covar(self, shape, add_noise=True, *params, **kwargs):
        if not self.has_task_noise:
            #data_noise = self.noise_covar(*params, shape=torch.Size((shape[:-2],)), **kwargs)
            data_noise = self.noise_covar(*params, shape=torch.Size((shape[-2],)), **kwargs)
            eye = torch.ones(1, device=data_noise.device, dtype=data_noise.dtype)
            # TODO: add in a shape for batched models
            task_noise = ConstantDiagLazyTensor(
                eye, diag_shape=torch.Size((self.num_tasks,))
            )
            # eye2 = torch.ones(torch.Size((self.num_tasks,)), device=data_noise.device, dtype=data_noise.dtype)
            # task_noise = DiagLazyTensor(eye2)
            
            return KroneckerProductLazyTensor(data_noise, task_noise)
        else:
            # TODO: copy over pieces from MultitaskGaussianLikelihood
            raise NotImplementedError("Task noises not supported yet.")
        
class FixedTaskNoiseMultitaskLikelihood2(_MultitaskGaussianLikelihoodBase):
    def __init__(self, data_noise, task_noise=None, *args, **kwargs):
        noise_covar = FixedGaussianNoise(noise=data_noise)
        super().__init__(noise_covar=noise_covar, *args, **kwargs)
        self.has_global_noise = False
        
        if task_noise is not None:
            self.has_task_noise = True
            self.task_noise = task_noise
        else:
            self.has_task_noise = False

        
    def _shaped_noise_covar(self, shape, add_noise=True, *params, **kwargs):
        if not self.has_task_noise:
            #data_noise = self.noise_covar(*params, shape=torch.Size((shape[:-2],)), **kwargs)
            data_noise = self.noise_covar(*params, shape=torch.Size((shape[-2],)), **kwargs)
            eye = torch.ones(1, device=data_noise.device, dtype=data_noise.dtype)
            # TODO: add in a shape for batched models
            # task_noise = ConstantDiagLazyTensor(
            #     eye, diag_shape=torch.Size((self.num_tasks,))
            # )
            eye2 = torch.ones(torch.Size((self.num_tasks,)), device=data_noise.device, dtype=data_noise.dtype)
            task_noise = DiagLazyTensor(eye2)
            
            return KroneckerProductLazyTensor(data_noise, task_noise)
        else:
            data_noise = self.noise_covar(*params, shape=torch.Size((shape[-2],)), **kwargs)
            eye = torch.tensor(self.task_noise, device=data_noise.device, dtype=data_noise.dtype)
            task_noise = DiagLazyTensor(eye)
            return KroneckerProductLazyTensor(data_noise, task_noise)


            # TODO: copy over pieces from MultitaskGaussianLikelihood
            #raise NotImplementedError("Task noises not supported yet.")


class FixedTaskNoiseMultitaskLikelihood_LinOP(_MultitaskGaussianLikelihoodBase):
    def __init__(self, noise, task_noise= None, *args, **kwargs):
        noise_covar = FixedGaussianNoise(noise=noise)
        super().__init__(noise_covar=noise_covar, *args, **kwargs)

        if task_noise is not None:
            self.has_task_noise = True
            self.task_noise = task_noise
        else:
            self.has_task_noise = False

        self.has_global_noise = False
        
    def _shaped_noise_covar(self, shape, add_noise=True, *params, **kwargs):
        if not self.has_task_noise:
            data_noise = self.noise_covar(*params, shape=torch.Size((shape[-2],)), **kwargs)
            eye = torch.ones(1, device=data_noise.device, dtype=data_noise.dtype)
            # TODO: add in a shape for batched models
            task_noise = operators.ConstantDiagLinearOperator(eye, diag_shape=torch.Size((self.num_tasks,)))
            return operators.KroneckerProductLinearOperator(data_noise, task_noise)
        else:
            data_noise = self.noise_covar(*params, shape=torch.Size((shape[-2],)), **kwargs)
            eye = torch.tensor(self.task_noise, device=data_noise.device, dtype=data_noise.dtype)
            # TODO: add in a shape for batched models
            task_noise = operators.ConstantDiagLinearOperator(eye, diag_shape=torch.Size((self.num_tasks,)))
            return operators.KroneckerProductLinearOperator(data_noise, task_noise)
            
            # TODO: copy over pieces from MultitaskGaussianLikelihood
            raise NotImplementedError("Task noises not supported yet.")
        
    def set_noise(self, noise):
        self.noise_covar = FixedGaussianNoise(noise=noise)
        return None

class MultitaskGaussianLikelihoodWithMissingObs(MultitaskGaussianLikelihood):
    '''
    from Andreas
    '''


    def __init__(self, num_tasks=None, noise_constraint=None, has_task_noise=True, original_shape=None):
        # Call super properly
        super().__init__(num_tasks=num_tasks, noise_constraint=noise_constraint, has_task_noise=has_task_noise)
        if original_shape is None:
            self.original_shape = None
        else:
            self.original_shape = original_shape

        # You are allowed to have as many noise parameters as you need
        # You will get this information according to the call in the jupyter
        # But you will get the way of calculating the _exact_ noise from the
        # GP Model through an additional helper function or something similar

        # Standard way: Simply flatten and mask according to the model
        self.noise_strategy = None


    def set_noise_strategy(self, noise_strategy):
        self.noise_strategy = noise_strategy
        return None

    #def __call__(self):
        """
        In `DefaultPredictionStrategy.__init__()` the line
        `mvn = self.likelihood(train_prior_dist, train_inputs)` is used
        to extract the `train_train_covar`.
        This variable is, in turn, used to calculate the `covar_cache` in
        the `exact_prediction`.
        This call should therefore return the masked `train_train_covar`
        to ensure the correct dimensions in the prediction calculation.
        """
        # Might even be unnecessary
    #    return None


    def marginal(self, function_dist: MultivariateNormal, *params: Any, **kwargs: Any) -> MultivariateNormal:
        """
        Adjust this call for the likelihood calculation of multivariate GPs
        with missing observations.
        That is, do masking and only consider the values that actually exist.
        """

        # TODO catch the case where mask is empty and fall back to the super()s behaviour
        



        # If params is not empty, is a call from the exact_prediction_strategy
        # (Potentially also some other function)




        # Extract mean and covar, same as super().marginal()
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix

        # The covariance matrix is assumed to be the right size
        # Add noise to the covar
        #noise_target_shape = covar.shape - self.original_shape
        #import pdb
        #pdb.set_trace()

        if self.training or (not self.training and not params == ()) or (not self.training and settings.prior_mode.on()):
            noise_matrix = super()._shaped_noise_covar(self.original_shape, add_noise=self.has_global_noise)
            result_covar = covar + self.noise_strategy(noise=self.noise, task_noises=self.task_noises, noise_matrix=noise_matrix)
        else:
            if isinstance(self.noise_strategy, ManualNoise) or isinstance(self.noise_strategy, MaskedManualNoise): 
                # I assume that kwargs["train_data"] and "current_data" exists
                train_x = kwargs["train_data"]
                train_x = train_x.reshape(train_x.shape[0])
                test_x = kwargs["current_data"]
                mask = kwargs["mask"]

                # How many of the training values exist in the test dataset?
                train_test_eq_vals = [any(torch.isclose(train_val, test_x)) for train_val in train_x]

                # Only used for duplicate training values, as cause anomalies in the remaining code
                # Main assumption: If a point exists multiple times in the training set, it has the same manual noise
                # If you don't have that, do something else, like adding 1e-18 to the point or something, that's your problem now
                
                dists = (torch.abs(train_x.unsqueeze(1) - train_x.unsqueeze(1).T) < 1e-8)
                duplicate_count = (dists.sum() - len(train_x))/2
                delete_counter = 0
                while duplicate_count > 0:
                    dists = (torch.abs(train_x.unsqueeze(1) - train_x.unsqueeze(1).T) < 1e-8).sum(dim=0)
                    for i in range(len(dists)):
                        if dists[i] > 1:
                            train_x = torch.cat((train_x[:i], train_x[i + 1:]))
                            delete_counter += 1
                            break
                    dists = (torch.abs(train_x.unsqueeze(1) - train_x.unsqueeze(1).T) < 1e-8)
                    duplicate_count = (dists.sum() - len(train_x))/2
                # train_x = unique_train_x 
                # There are some equal values
                if any(train_test_eq_vals):
                    # Make out the locations where these values are and which train noise to insert
                    # This _should_ contain the indices
                    # Check if a test point occurs multiple times in the training dataset (i.e. "True"s at the non-diagonals)

                    test_train_eq_vals = [torch.isclose(test_val, train_x).tolist() + [False]*delete_counter for test_val in test_x]

                    # Inflate the equality check results to the number of tasks
                    inflated_test_train_eq_vals = [[val for val in dimension for _ in range(self.num_tasks)] for dimension in test_train_eq_vals]
                    # This is about getting the manual noise
                    noise_matrix = super()._shaped_noise_covar(self.original_shape, add_noise=self.has_global_noise)
                    result_noise = self.noise_strategy(noise=self.noise, task_noises=self.task_noises, noise_matrix=noise_matrix, eval_mode=True)
                    result_noise = torch.diag(result_noise.diag() * (~mask))
                    # Sum the noise value lists to be one big tensor containing only 1s and 0s
                    # Multiply with the diagonal to filter, where the noise is not necessary there should be 0
                    #filtered_result_noise = (result_noise.diag() * torch.Tensor(inflated_test_train_eq_vals)).sum(0)
                    filtered_result_noise = torch.tensor([ result_noise.diag()[inflated_test_train_eq_val].tolist() if any(inflated_test_train_eq_val) else torch.zeros(self.num_tasks).tolist() for inflated_test_train_eq_val in inflated_test_train_eq_vals]).flatten() 
                    # Sanity check: The noise from super() and my noise should have the same size
                    classical_noise = super()._shaped_noise_covar((int(mean.numel()/self.num_tasks), self.num_tasks), add_noise=self.has_global_noise)
                    if not len(filtered_result_noise) == len(classical_noise.diag()):
                        raise "Wrong lengths, something was filtered wrong"
                    filtered_classical_noise =  classical_noise.diag() * torch.tensor([all(~torch.tensor(test_train_eq_val)) for test_train_eq_val in test_train_eq_vals for _ in range(self.num_tasks)])
                    total_noise = filtered_result_noise + filtered_classical_noise 
                    result_covar = covar + torch.diag(total_noise)

            else:
                result_covar = covar + super()._shaped_noise_covar((int(mean.numel()/self.num_tasks), self.num_tasks), add_noise=self.has_global_noise)



        return function_dist.__class__(mean, result_covar)
