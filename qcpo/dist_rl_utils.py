
import numpy as np
import torch

from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.misc import zeros
from rlpyt.utils.logging import logger


def normalize(value, valid=None):
    if valid is not None:
        valid_mask = valid > 0
        mean = value[valid_mask].mean()
        std = value[valid_mask].std()
    else:
        mean = value.mean()
        std = value.std()
    return (value - mean) / max(std, 1e-6)


def quantile_huber_loss(quantiles, target_quantiles, tau, valid):
    pairwise_delta = target_quantiles[:, :, None, :] - quantiles[:, :, :, None]  # T x B x n_quantile x n_target_quantile
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss


def weibull_tail_loss(quantiles, c_w_alpha, c_w_beta, c_tau, tail_ind, valid):
    quantiles_t = torch.sort(quantiles)[0].detach()
    loss = (0.5 * (torch.log(c_w_beta)[:, :, None] + c_w_alpha.reciprocal()[:, :, None] * torch.log(c_tau[None, None, tail_ind:]) - torch.log(quantiles_t[:, :, tail_ind:])) ** 2).mean()
    return loss


@torch.no_grad()
def quantile_target_estimation(reward, r_dist, done, bootstrap_r_dist, discount):
    """Time-major inputs, optional other dimensions: [T], [T,B], etc.  Similar
    to `discount_return()` but using Generalized Advantage Estimation to
    compute advantages and returns."""

    quantile_target = zeros(r_dist.shape, dtype=r_dist.dtype)
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd
    quantile_target[-1] = reward[-1].unsqueeze(1) + discount * bootstrap_r_dist.squeeze(0) * nd[-1].unsqueeze(1)
    for t in reversed(range(len(reward) - 1)):
        quantile_target[t] = reward[t].unsqueeze(1) + discount * r_dist[t + 1] * nd[t].unsqueeze(1)
    return quantile_target


@torch.no_grad()
def gae_quantile_simple(reward, r_dist, done, r_bdist, discount, gae_lambda):
    r_dist_t = torch.sort(r_dist)[0]
    r_bdist_t = torch.sort(r_bdist)[0]

    advantage_quantile = zeros(r_dist.shape, dtype=r_dist.dtype)
    return_quantile = zeros(r_dist.shape, dtype=r_dist.dtype)

    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd

    advantage_quantile[-1] = reward[-1][:, None] + nd[-1][:, None] * discount * r_bdist_t - r_dist_t[-1]
    for t in reversed(range(len(reward) - 1)):
        delta = reward[t][:, None] + nd[t][:, None] * discount * r_dist_t[t+1] - r_dist_t[t]
        advantage_quantile[t] = delta + discount * gae_lambda * nd[t][:, None] * advantage_quantile[t + 1]
    return_quantile[:] = advantage_quantile + r_dist_t
    return advantage_quantile, return_quantile


@torch.no_grad()
def compute_prob_ratio(cost, c_weibull_tail, c_w_alpha, c_w_beta, done, c_bw_alpha, c_bw_beta, discount, log_clip_range=0.2, normalize=True):
    EPS = 1e-3
    nd = 1 - done
    nd = nd.type(cost.dtype) if isinstance(nd, torch.Tensor) else nd

    c_w_alpha_next = torch.cat((c_w_alpha[1:], c_bw_alpha), 0)
    c_w_beta_next = torch.cat((c_w_beta[1:], c_bw_beta), 0)

    c_weibull_tail_target = (c_weibull_tail - cost[:, :, None]) / discount
    possible_ind = (c_weibull_tail_target > EPS)
    c_weibull_tail_target = torch.clamp(c_weibull_tail_target, min=EPS)

    log_prob_ratio = torch.zeros_like(c_weibull_tail)
    log_prob_c_weibull_tail = torch.log(c_w_alpha)[:, :, None] - c_w_alpha[:, :, None] * torch.log(c_w_beta)[:, :, None] \
                              + (c_w_alpha - 1)[:, :, None] * torch.log(c_weibull_tail) \
                              - torch.pow(c_weibull_tail / c_w_beta[:, :, None], c_w_alpha[:, :, None])
    log_prob_c_weibull_tail_target = torch.log(c_w_alpha_next)[:, :, None] - c_w_alpha_next[:, :, None] * torch.log(c_w_beta_next)[:, :, None] \
                                     +(c_w_alpha_next - 1)[:, :, None] * torch.log(c_weibull_tail_target) \
                                     - torch.pow(c_weibull_tail_target / c_w_beta_next[:, :, None], c_w_alpha_next[:, :, None])

    log_prob_ratio[possible_ind] = (log_prob_c_weibull_tail_target - torch.log(torch.tensor(discount)) - log_prob_c_weibull_tail)[possible_ind]

    # Normalize log_prob_ratio
    if normalize:
        avg_log = torch.mean(log_prob_ratio, dim=[0, 1])
        std_log = torch.std(log_prob_ratio, dim=[0, 1])
        log_prob_ratio = (log_prob_ratio - avg_log[None, None, :]) / (5.0 * std_log[None, None, :])

    log_prob_ratio = nd[:, :, None] * torch.clamp(log_prob_ratio, min=-log_clip_range, max=log_clip_range)
    prob_ratio = torch.exp(log_prob_ratio)

    return prob_ratio




