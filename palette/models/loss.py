from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def mse_loss(pred: torch.Tensor, true: torch.Tensor):
    """ Mean squared error loss """
    return F.mse_loss(pred, true)


class FocalLoss(nn.Module):
    def __init__(self, gamma: int = 2, alpha: float | List | None = None, size_average: bool = True):
        """
        Focal Loss for addressing class imbalance.
        Args:
            gamma (int): Focusing parameter for modulating factor (1-p).
            alpha (float or list or None): Weighting factor for classes.
            size_average (bool): Whether to average the loss.
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, pred: torch.Tensor, true: torch.Tensor):
        """
        Forward pass for Focal Loss
        """
        if pred.dim() > 2:
            pred = pred.view(pred.size(0), pred.size(1), -1) # N,C,H,W => N,C,H*W
            pred = pred.transpose(1, 2)                      #            N,H*W,C
            pred = pred.contiguous().view(-1, pred.size(2))  #            N*H*W,C
        true = true.view(-1, 1)

        logpt = F.log_softmax(pred)
        logpt = logpt.gather(1, true)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != pred.data.type():
                self.alpha = self.alpha.type_as(pred.data)
            at = self.alpha.gather(0, true.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

