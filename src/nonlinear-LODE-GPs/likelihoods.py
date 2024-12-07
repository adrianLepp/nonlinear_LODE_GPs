from gpytorch.likelihoods.multitask_gaussian_likelihood import _MultitaskGaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import FixedGaussianNoise
from gpytorch.lazy import ConstantDiagLazyTensor, KroneckerProductLazyTensor, DiagLazyTensor
import torch

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
            # task_noise = ConstantDiagLazyTensor(
            #     eye, diag_shape=torch.Size((self.num_tasks,))
            # )
            eye2 = torch.ones(torch.Size((self.num_tasks,)), device=data_noise.device, dtype=data_noise.dtype)
            task_noise = DiagLazyTensor(eye2)
            
            return KroneckerProductLazyTensor(data_noise, task_noise)
        else:
            # TODO: copy over pieces from MultitaskGaussianLikelihood
            raise NotImplementedError("Task noises not supported yet.")