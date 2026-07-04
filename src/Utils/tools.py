"""
    This code implements some useful functions to train
    GNN models for infection risk prediction on graph-based
    datasets
"""
import os
import sys
import psutil # To monitor CPU memory usage
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.special import digamma
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

def get_process_cpu_memory():
    """Returns memory usage of the current process in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2  # in MB


# This following re-implementation of the forward function of the STM model intends to monitor CPU memory usage
def forward(model, xt, ids, edge_index, edge_attr, timestamps):
    """Forward pass of the STM model."""
    if xt is None or len(xt) == 0:
        return torch.empty(0, model.d_h, device=model.device) if not model.hetero_mode else {k: torch.empty(0, model.d_h, device=model.device) for k in xt}

    # Temporal encoding
    x = model.add_time_encoding_(xt, timestamps) 
    x = {k: model.proj_layer[k](x[k]) for k in x} if model.hetero_mode else model.proj_layer(x)

    # Spatial embeddings
    xs = model.augments_spatial_(x, ids)
    spatial_embeddings = model.spatial_module(xs, ids, edge_index, edge_attr, timestamps)
    print("\t=======>CPU memory usage before mem_spatial update: {} Mb".format(get_process_cpu_memory()))
    model.mem_spatial_(spatial_embeddings, ids)
    print("\t=======>CPU memory usage after mem_spatial update: {} Mb".format(get_process_cpu_memory()))

    # Temporal embeddings
    xt = model.augments_temporal_(x, ids)
    temporal_embeddings = model.temporal_module(xt, ids, edge_index, edge_attr, timestamps)
    print("\t=======>CPU memory usage before mem_temporal update: {} Mb".format(get_process_cpu_memory()))
    model.mem_temporal_(temporal_embeddings, ids)
    print("\t=======>CPU memory usage after mem_temporal update: {} Mb".format(get_process_cpu_memory()))

    return model.embed_(xt, spatial_embeddings, temporal_embeddings)


# Function to create a sliding window from a set of indices
def sliding_windows(idx_list, windows_size, stride=1):
    #return [idx_list[i:i + windows_size] for i in range(0, len(idx_list) - windows_size + 1, stride)] # Does not include partial windows (when remaining elements are fewer than windows_size)
    return [idx_list[i:i + windows_size] for i in range(0, len(idx_list), stride)] # Includes partial windows (when remaining elements are fewer than windows_size)

def one_hot_encoding_np(targets, num_classes=4):
    """
        Encodes an array of integers into a one-hot
        encoding based on the number of classes.

        Parameters:
        -----------
        targets_np: np.array
            Array of INTEGERS that we want to encode.
        num_classes: int
            Number of classes to use for the one-hot encoding

        Returns:
        --------
        one_hot_targets: np.array
            Array of shape (targets.shape[0], num_classes)
            containing the one-hot encodings
    """
    one_hot_targets = np.eye(num_classes)[targets]
    
    return one_hot_targets


# Version used for the experiments previous to 07/11/2025
def get_uncertainties_OLD(dirichlet_alpha):
    """
        Computes the epistemic, aleatoric and total uncertainties
        for the output of an evidential learning Dirichlet-based
        classificaiton model.

        Parameters:
        -----------
        dirichlet_alpha: torch.tensor
            Alphas of the Dirichlet distribution, prediction of 
            a Dirichlet-based evidential learning model. It is 
            of shape (batch_size, n_classes)

        Returns:
        --------
        epistemic: torch.Tensor
            Epistemic (model-dependent and reducible) uncertainty.
            It ranges between 0 and 1, and smaller values are better.
        aleatoric: torch.Tensor
            Aleatoric (data-dependent and non-reducible) uncertainty.
            It ranges between 0 and log_e(K) where K is the number of
            classes (for 4 classes log_e(K) ~ 1.386). Smaller values are better.
        total: torch.Tensor
            Total uncertainty.
            It ranges between 0 and log_e(K) where K is the number of
            classes (for 4 classes log_e(K) ~ 1.386). Smaller values are better.
    """
    K = dirichlet_alpha.size(1)
    S = torch.sum(dirichlet_alpha, dim=1, keepdim=True)
    probs = dirichlet_alpha / S

    # Epistemic: high when total evidence is low
    epistemic = K / S.squeeze()

    # Aleatoric: expected entropy under Dirichlet
    psi_alpha_plus_1 = digamma(dirichlet_alpha + 1)
    psi_S_plus_1 = digamma(S + 1)
    aleatoric = -torch.sum(probs * (psi_alpha_plus_1 - psi_S_plus_1), dim=1)

    # Total: entropy of the mean
    total = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

    return epistemic, aleatoric, total


def get_uncertainties(dirichlet_alpha):
    """
        Computes the epistemic, aleatoric and total uncertainties
        for the output of an evidential learning Dirichlet-based
        classificaiton model.

        IMPORTANT: The formulas for the quantification of epistemic
        and aleatoric uncertainties come from equations 21 and 23
        of the following paper: https://journals.ametsoc.org/view/journals/aies/3/4/AIES-D-23-0093.1.xml

        Parameters:
        -----------
        dirichlet_alpha: torch.tensor
            Alphas of the Dirichlet distribution, prediction of 
            a Dirichlet-based evidential learning model. It is 
            of shape (batch_size, n_classes)

        Returns:
        --------
        epistemic: torch.Tensor
            Epistemic (model-dependent and reducible) uncertainty.
            It ranges between 0 and 1, and smaller values are better.
        aleatoric: torch.Tensor
            Aleatoric (data-dependent and non-reducible) uncertainty.
            It ranges between 0 and log_e(K) where K is the number of
            classes (for 4 classes log_e(K) ~ 1.386). Smaller values are better.
        total: torch.Tensor
            Total uncertainty.
            It ranges between 0 and log_e(K) where K is the number of
            classes (for 4 classes log_e(K) ~ 1.386). Smaller values are better.
    """
    # Getting number of samples N and number of classes K
    N = dirichlet_alpha.size(0)
    K = dirichlet_alpha.size(1)

    # Getting the sum of alphas per sample
    S = torch.sum(dirichlet_alpha, dim=1, keepdim=True)

    # Getting the probability score for each sample and class
    probs = dirichlet_alpha / S

    # Total uncertainty
    total = probs - probs**2 # Obtained by summing Eqs. 21 and 23 of https://journals.ametsoc.org/view/journals/aies/3/4/AIES-D-23-0093.1.xml

    # Epistemic uncertainty
    epistemic = probs*(1-probs)/(S+1)

    # Aleatoric uncertainty
    aleatoric = total - epistemic

    return epistemic, aleatoric, total
