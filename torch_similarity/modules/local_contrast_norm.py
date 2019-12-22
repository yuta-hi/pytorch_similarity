from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from .gradient_difference import _gauss_param
from ..functional import local_contrast_norm_nd

class LocalContrastNorm1d(torch.nn.Module):
    """ One-dimensional local contrast normalization (LCN)

    Args:
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to 3.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def __init__(self,
                 gauss_sigma=3,
                 gauss_truncate=4.0,
                 eps=1e-8):

        super(LocalContrastNorm1d, self).__init__()

        self.gauss_sigma = gauss_sigma
        self.gauss_truncate = gauss_truncate
        self.gauss_kernel = None

        self.eps = eps

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        self.gauss_kernel = _gauss_param(1, self.gauss_sigma, self.gauss_truncate)

    def _freeze_params(self):
        self.gauss_kernel.requires_grad = False

    def _check_type_forward(self, x):
        if x.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(x.dim()))

    def forward(self, x):
        self._check_type_forward(x)
        self._freeze_params()
        return local_contrast_norm_nd(x, self.gauss_kernel, self.eps)


class LocalContrastNorm2d(LocalContrastNorm1d):
    """ Two-dimensional local contrast normalization (LCN)

    Args:
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to 3.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def __init__(self,
                 gauss_sigma=3,
                 gauss_truncate=4.0,
                 eps=1e-8):

        super(LocalContrastNorm2d, self).__init__(
                gauss_sigma, gauss_truncate, eps)

    def _initialize_params(self):
        self.gauss_kernel = _gauss_param(2, self.gauss_sigma, self.gauss_truncate)

    def _check_type_forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))


class LocalContrastNorm3d(LocalContrastNorm1d):
    """ Three-dimensional local contrast normalization (LCN)

    Args:
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to 3.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def __init__(self,
                 gauss_sigma=3,
                 gauss_truncate=4.0,
                 eps=1e-8):

        super(LocalContrastNorm3d, self).__init__(
                gauss_sigma, gauss_truncate, eps)

    def _initialize_params(self):
        self.gauss_kernel = _gauss_param(3, self.gauss_sigma, self.gauss_truncate)

    def _check_type_forward(self, x):
        if x.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(x.dim()))
