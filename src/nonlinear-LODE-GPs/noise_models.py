import torch

def mask_noise(mask, noise=None, task_noises=None, noise_matrix=None):
    """
    Create a constant noise matrix of the size of the mask
    """
    if noise is None and task_noises is None:
        return torch.zeros((int(torch.sqrt(mask.count_nonzero())), int(torch.sqrt(mask.count_nonzero()))))
    else:
        if not noise_matrix is None:
            full_noise = noise_matrix
        else:
            full_noise = torch.zeros((int(torch.sqrt(mask.count_nonzero())), int(torch.sqrt(mask.count_nonzero()))))
        height, width = full_noise.shape
        mask = mask.broadcast_to((height, width))
        mask = ~(mask | mask.T)
        if not isinstance(full_noise, torch.Tensor):
            temp = full_noise.evaluate()[mask]
        else:
            temp = full_noise[mask]
        masked_full_noise = temp.reshape((int(torch.sqrt(mask.count_nonzero())), int(torch.sqrt(mask.count_nonzero()))))
        return masked_full_noise

def manual_noise(manual_noise, noise=None, task_noises=None, noise_matrix=None):
    if noise is None and task_noises is None:
        return torch.diag(manual_noise)
    else:
        if not noise_matrix is None:
            full_noise = noise_matrix
            if any(manual_noise.isnan().flatten()):
                manual_noise_mask = manual_noise.flatten().isnan()
                noise_matrix[~manual_noise_mask] = torch.ones_like(noise_matrix)[~manual_noise_mask] * torch.diag(manual_noise)
            else:
                full_noise = torch.diag(manual_noise)
        else:
            full_noise = torch.diag(manual_noise)
        height, width = full_noise.shape
        # This can die
        temp = full_noise
        masked_full_noise = temp.reshape((height, width))
        return masked_full_noise



class MaskedNoise():

    def __init__(self, mask):
        """
        Creates a masked noise strategy based on the given mask.
        """
        self.mask = mask

    def __call__(self, noise=None, task_noises=None, noise_matrix=None):
        """
        Create a constant noise matrix of the size of the mask
        """
        return mask_noise(self.mask, noise, task_noises, noise_matrix)


class ManualNoise():

    def __init__(self, manual_noise):
        self.manual_noise = manual_noise


    def __call__(self, noise, task_noises, noise_matrix, man_noise=None):
        if man_noise is None:
            return manual_noise(self.manual_noise, noise, task_noises, noise_matrix)
        else:
            return manual_noise(man_noise, noise, task_noises, noise_matrix)




class MaskedManualNoise():
    """
    Deals with masked noise (i.e. missing observations) and manually setting some noise points.
    """

    def __init__(self, mask, manual_noise):
        self.mask = mask
        self.manual_noise = manual_noise
        return None

    def __call__(self, noise=None, task_noises=None, noise_matrix=None, eval_mode=False):
        manual_masked_noise = manual_noise(self.manual_noise, noise, task_noises, noise_matrix=noise_matrix)
        # Mask the noise
        if not eval_mode:
            masked_noise = mask_noise(self.mask, noise, task_noises, manual_masked_noise)
            # Now replace the values I want to manually choose
            return masked_noise
        else:
            return manual_masked_noise
