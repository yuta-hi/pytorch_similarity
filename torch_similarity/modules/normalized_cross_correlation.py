from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..functional import normalized_cross_correlation

class NormalizedCrossCorrelation(nn.Module):
    """ N-dimensional normalized cross correlation (NCC)

    Args:
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self,
                 eps=1e-8,
                 return_map=False,
                 reduction='mean'):

        super(NormalizedCrossCorrelation, self).__init__()

        self._eps = eps
        self._return_map = return_map
        self._reduction = reduction

    def forward(self, x, y):

        return normalized_cross_correlation(x, y,
                            self._return_map, self._reduction, self._eps)
