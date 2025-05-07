from functools import wraps
from time import time
import einops

import torch
import torch.nn as nn


def timing(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        tic = time.time()
        res = f(**args, **kwargs)
        tac = time.time()
        print(f'function {f.__name__} took {tac-tic:2.4f} seconds')
        return res
    return wrap


class weighted_mse:
    def __init__(self, scaling=90):
        self.mse = nn.MSELoss(reduction="none")
        self.scaling = scaling

    def __call__(self, inp, target):
        # get target weight vector
        weights = torch.ones(target.shape, device=inp.device)
        weights[target > 0] = self.scaling
        return (weights * self.mse(inp, target)).mean()


def binary_metrics(pred, lab, link_threshold, p_value=False):
    if not p_value:
        binary = pred > link_threshold
    else:
        binary = pred < link_threshold

    tp = torch.sum((binary == 1) * (lab == 1))
    tn = torch.sum((binary == 0) * (lab == 0))
    fp = torch.sum((binary == 1) * (lab == 0))
    fn = torch.sum((binary == 0) * (lab == 1))
    assert torch.all(tp + fp + tn + fn), "BROKEN metric"
    # recall / sensitivity / true positive rate - false positive rate - true negative rate - false negative rate
    return tp / (tp + fn), fp / (fp + tn), tn / (fp + tn), fn / (tp + fn)


def lagged_batch_corr(points, max_lags):
    # calculates the autocovariance matrix with a batch dimension
    # lagged variables are concated in the same dimension.
    # inpuz (B, time, var)
    # roll to calculate lagged cov:
    B, N, D = points.size()

    # we roll the data and add it together to have the lagged versions in the table
    stack = torch.concat(
        [torch.roll(points, x, dims=1) for x in range(max_lags + 1)], dim=2
    )

    mean = stack.mean(dim=1).unsqueeze(1)
    std = stack.std(dim=1).unsqueeze(1) + 1e-6 # avoiding division by zero
    diffs = (stack - mean).reshape(B * N, D * (max_lags + 1))

    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(
        B, N, D * (max_lags + 1), D * (max_lags + 1)
    )

    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    # make correlation out of it by dividing by the product of the stds
    corr = bcov / (
        std.repeat(1, D * (max_lags + 1), 1).reshape(
            std.shape[0], D * (max_lags + 1), D * (max_lags + 1)
        )
        * std.permute((0, 2, 1))
    )
    # we can remove backwards in time links. (keep only the original values)
    return corr[:, :D, D:]  # (B, D, D)


def transform_corr_to_y(corr, ml, n_vars):
    ncorr = einops.rearrange(corr[:, :n_vars:], "b c1 (t c2) -> b c1 c2 t", t=ml)
    fncorr = torch.flip(ncorr, dims=[3])
    return fncorr


def custom_corr_regularization(predictions, data, exp=1.5, epsilon=0.15):
    # predictions: (batch.caused, causing, lag)
    # data: batch, t, n_vars
    # penalized predictions if the cov of the corresponding link is low.
    ml = predictions.shape[3]
    n_vars = data.shape[2]

    # rashape everything properly
    corr = lagged_batch_corr(data, ml)
    fncorr = transform_corr_to_y(corr, ml, n_vars)
    # specifying the batch size
    regularization = 1 / (torch.abs(fncorr) + epsilon)  # for numeric stability
    penalty = torch.mean((predictions * regularization) ** exp)
    return penalty