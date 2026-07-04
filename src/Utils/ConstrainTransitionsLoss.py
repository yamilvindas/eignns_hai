import torch
import torch.nn as nn
"""
    Implemented with the help of ChatGPT.
"""

# y_pred: [batch_size, num_classes], logits or probabilities
# mask: [batch_size, num_classes], 0 for impossible, 1 for possible

def ConstrainTransitionsLoss(y_pred, mask):
    # MSE to enforce zero on impossible classes
    loss_zero = ((y_pred * (1 - mask))**2).mean()  # enforces 0 where mask=0

    return loss_zero
