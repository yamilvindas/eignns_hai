#!/usr/bin/env python3
"""
    Code STRONGLY INSPIRED from: https://github.com/HanxunH/SCELoss-Reproduce/blob/master/loss.py
"""

import torch
import torch.nn.functional as F

class SCELoss(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=3, device=torch.device('cpu'), class_weights=None, is_soft_labels=False):
        super(SCELoss, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.class_weights = class_weights
        if (self.class_weights is None):
            self.cross_entropy = torch.nn.CrossEntropyLoss()
        else:
            self.cross_entropy = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.is_soft_labels = is_soft_labels

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        if (not self.is_soft_labels):
            soft_label = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        else:
            soft_label = labels
        soft_label = torch.clamp(soft_label, min=1e-4, max=1.0)
        rce = pred.to(self.device) * torch.log(soft_label.to(self.device))
        # Applying class weights to RCE
        if (self.class_weights is not None):
            # Comment the following two lines if the previous lines are uncommented
            if (type(self.class_weights) != torch.Tensor):
                self.class_weights = torch.tensor(self.class_weights)
            self.class_weights = self.class_weights.to(self.device)
            rce = self.class_weights*rce

        rce = -1*torch.sum(rce, dim=1)

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()

        return loss
