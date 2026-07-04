#!/usr/bin/env python3
"""
    Modification of the original evidential_classification loss
    from edl_pytorch to add class weights: https://github.com/teddykoker/evidential-learning-pytorch/blob/main/edl_pytorch/loss.py
"""
from copy import deepcopy
import torch
import torch.nn.functional as F

# Normal Inverse Gamma Negative Log-Likelihood
# from https://arxiv.org/abs/1910.02600:
# > we denote the loss, L^NLL_i as the negative logarithm of model
# > evidence ...
def nig_nll(gamma, v, alpha, beta, y, per_sample=False):
    two_beta_lambda = 2 * beta * (1 + v)
    t1 = 0.5 * (torch.pi / v).log()
    t2 = alpha * two_beta_lambda.log()
    t3 = (alpha + 0.5) * (v * (y - gamma) ** 2 + two_beta_lambda).log()
    t4 = alpha.lgamma()
    t5 = (alpha + 0.5).lgamma()
    nll = t1 - t2 + t3 + t4 - t5
    if (per_sample):
        return nll 
    else:
        return nll.mean()


# Normal Inverse Gamma regularization
# from https://arxiv.org/abs/1910.02600:
# > we formulate a novel evidence regularizer, L^R_i
# > scaled on the error of the i-th prediction
def nig_reg(gamma, v, alpha, _beta, y, per_sample=False):
    reg = (y - gamma).abs() * (2 * v + alpha)
    if (per_sample):
        return reg 
    else:
        return reg.mean()


# KL divergence of predicted parameters from uniform Dirichlet distribution
# from https://arxiv.org/abs/1806.01768
# code based on:
# https://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
def dirichlet_reg(alpha, y, per_sample=False):
    # dirichlet parameters after removal of non-misleading evidence (from the label)
    alpha = y + (1 - y) * alpha

    # uniform dirichlet distribution
    beta = torch.ones_like(alpha)

    sum_alpha = alpha.sum(-1)
    sum_beta = beta.sum(-1)

    t1 = sum_alpha.lgamma() - sum_beta.lgamma()
    t2 = (alpha.lgamma() - beta.lgamma()).sum(-1)
    t3 = alpha - beta
    t4 = alpha.digamma() - sum_alpha.digamma().unsqueeze(-1)

    kl = t1 - t2 + (t3 * t4).sum(-1)

    if (per_sample):
        return kl 
    else:
        return kl.mean()


# Eq. (5) from https://arxiv.org/abs/1806.01768:
# Sum of squares loss
def dirichlet_mse(alpha, y, per_sample=False):
    sum_alpha = alpha.sum(-1, keepdims=True)
    p = alpha / sum_alpha
    t1 = (y - p).pow(2).sum(-1)
    t2 = ((p * (1 - p)) / (sum_alpha + 1)).sum(-1)
    mse = t1 + t2
    if (per_sample):
        return mse 
    else:
        return mse.mean()

class EvidentialClassification(torch.nn.Module):
    """
        Adaptation of edl_pytorch to add class weights
    """

    def __init__(self, lamb: float = 1.0, class_weights = None) -> None:
        super().__init__()
        self.lamb = lamb
        self.epsilon = 1e-9
        self.class_weights = class_weights

    def forward(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Number of classes
        num_classes = alpha.shape[-1]

        # Getting the class weights vector if possible
        if (self.class_weights is not None):
            device_to_use = alpha.device
            class_weight_vector = torch.tensor([self.class_weights[true_class] for true_class in y]).to(device_to_use)

        # One-hot encoding the true labels
        #y_original = deepcopy(y).detach().cpu().numpy()
        y = F.one_hot(y, num_classes)

        # Getting the MSE loss per sample
        mse_per_sample = dirichlet_mse(alpha, y, per_sample=True)

        # Getting the regularization term per sample
        reg_per_sample = dirichlet_reg(alpha, y, per_sample=True)

        # Getting the final loss
        if (self.class_weights is not None):
            # IMPORTANT: ONLY MULTIPLY mse_per_sample BY class_weight_vector as 
            # IMPORTANT: it adjusts for class imbalance in the classification without scaling the regularization term.
            loss = (class_weight_vector*mse_per_sample).mean() + self.lamb * reg_per_sample.mean()
        else:
            loss = mse_per_sample.mean() + self.lamb * reg_per_sample.mean()

        return torch.mean(loss)