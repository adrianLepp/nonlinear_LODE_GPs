import torch
from gpytorch.lazy import lazify

def create_mask(train_labels : torch.Tensor):
    base_mask = train_labels.flatten().isnan()

    filtered_labels = train_labels.flatten()[~base_mask]

    return filtered_labels, base_mask




def masking(base_mask=None, train_labels=None, mean=None, covar=None, mvn=None, fill_zeros=False):
    # This is a crap call, should not have happened
    if base_mask == None and train_labels == None:
        raise "That's not how you use this!"
    elif base_mask == None and not train_labels == None:
        # I need to create a mask first
        base_mask = train_labels.flatten().isnan()
        filtered_labels = train_labels.flatten()[~base_mask]
    # invisible else assumes that base_mask exists and uses that

    if mean == None and covar == None and mvn == None:
        return filtered_labels, base_mask
    # This is a call with a MultivariateNormal
    elif mean == None and covar == None:
        mean, covar = mvn.loc, mvn.lazy_covariance_matrix
    if not mean.numel() == base_mask.numel():
        if fill_zeros:
            m = base_mask.tolist()
            m.extend([False]*(mean.numel()-base_mask.numel()))
            mask = torch.Tensor(m)
            base_mask = mask.bool()
        else:
            raise "Size doesn't match and fill_zero is False"
    # This is a call with a given mean and covar (usually from forward)
    full_mean = mean.flatten()[~base_mask]
    height, width = covar.evaluate().shape

    mask = base_mask.broadcast_to((height, width))
    mask = ~(mask | mask.T)
    temp = covar.evaluate()[mask]
    full_covar = temp.reshape((height-base_mask.count_nonzero(),width-base_mask.count_nonzero()))

    if train_labels == None:
        return full_mean, lazify(full_covar)
    else:
        return filtered_labels, full_mean, lazify(full_covar)