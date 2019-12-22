from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from ..functional import spatial_filter_nd
from .._helper import gauss_kernel_1d
from .._helper import gauss_kernel_2d
from .._helper import gauss_kernel_3d
from .._helper import gradient_kernel_1d
from .._helper import gradient_kernel_2d
from .._helper import gradient_kernel_3d

def _pair(x):
    if hasattr(x, '__getitem__'):
        return x
    return [x, x]

def _grad_param(ndim, method, axis):

    if ndim == 1:
        kernel = gradient_kernel_1d(method)
    elif ndim == 2:
        kernel = gradient_kernel_2d(method, axis)
    elif ndim == 3:
        kernel = gradient_kernel_3d(method, axis)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())

def _gauss_param(ndim, sigma, truncate):

    if ndim == 1:
        kernel = gauss_kernel_1d(sigma, truncate)
    elif ndim == 2:
        kernel = gauss_kernel_2d(sigma, truncate)
    elif ndim == 3:
        kernel = gauss_kernel_3d(sigma, truncate)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.Tensor(kernel).float())

class GradientDifference1d(nn.Module):
    """ One-dimensional gradient difference

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean'):

        super(GradientDifference1d, self).__init__()

        self.grad_method = grad_method
        self.gauss_sigma = _pair(gauss_sigma)
        self.gauss_truncate = gauss_truncate

        self.grad_kernel = None

        self.gauss_kernel_x = None
        self.gauss_kernel_y = None

        self.return_map = return_map
        self.reduction = reduction

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        self._initialize_grad_kernel()
        self._initialize_gauss_kernel()

    def _initialize_grad_kernel(self):
        self.grad_kernel = _grad_param(1, self.grad_method, axis=0)

    def _initialize_gauss_kernel(self):
        if self.gauss_sigma[0] is not None:
            self.gauss_kernel_x = _gauss_param(1, self.gauss_sigma[0], self.gauss_truncate)
        if self.gauss_sigma[1] is not None:
            self.gauss_kernel_y = _gauss_param(1, self.gauss_sigma[1], self.gauss_truncate)

    def _check_type_forward(self, x):
        if x.dim() != 3:
            raise ValueError('expected 3D input (got {}D input)'.format(x.dim()))

    def _freeze_params(self):
        self.grad_kernel.requires_grad = False
        if self.gauss_kernel_x is not None:
            self.gauss_kernel_x.requires_grad = False
        if self.gauss_kernel_y is not None:
            self.gauss_kernel_y.requires_grad = False

    def forward(self, x, y):

        self._check_type_forward(x)
        self._check_type_forward(y)
        self._freeze_params()

        if x.shape[1] != y.shape[1]:
            x = torch.mean(x, dim=1, keepdim=True)
            y = torch.mean(y, dim=1, keepdim=True)

        # reshape
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]

        x = x.view(b*c, 1, *spatial_shape)
        y = y.view(b*c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            x = spatial_filter_nd(x, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            y = spatial_filter_nd(y, self.gauss_kernel_y)

        # gradient magnitude
        x_grad = torch.abs(spatial_filter_nd(x, self.grad_kernel))
        y_grad = torch.abs(spatial_filter_nd(y, self.grad_kernel))

        # absolute difference
        diff = torch.abs(x_grad - y_grad)

        # reshape back
        diff_map = diff.view(b, c, *spatial_shape)

        if self.reduction == 'mean':
            diff = torch.mean(diff_map)
        elif self.reduction == 'sum':
            diff = torch.sum(diff_map)
        else:
            raise KeyError('unsupported reduction type: %s' % self.reduction)

        if self.return_map:
            return diff, diff_map

        return diff


class GradientDifference2d(nn.Module):
    """ Two-dimensional gradient difference

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean'):

        super(GradientDifference2d, self).__init__()

        self.grad_method = grad_method
        self.gauss_sigma = _pair(gauss_sigma)
        self.gauss_truncate = gauss_truncate

        self.grad_u_kernel = None
        self.grad_v_kernel = None

        self.gauss_kernel_x = None
        self.gauss_kernel_y = None

        self.return_map = return_map
        self.reduction = reduction

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        self._initialize_grad_kernel()
        self._initialize_gauss_kernel()

    def _initialize_grad_kernel(self):
        self.grad_u_kernel = _grad_param(2, self.grad_method, axis=0)
        self.grad_v_kernel = _grad_param(2, self.grad_method, axis=1)

    def _initialize_gauss_kernel(self):
        if self.gauss_sigma[0] is not None:
            self.gauss_kernel_x = _gauss_param(2, self.gauss_sigma[0], self.gauss_truncate)
        if self.gauss_sigma[1] is not None:
            self.gauss_kernel_y = _gauss_param(2, self.gauss_sigma[1], self.gauss_truncate)

    def _check_type_forward(self, x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def _freeze_params(self):
        self.grad_u_kernel.requires_grad = False
        self.grad_v_kernel.requires_grad = False
        if self.gauss_kernel_x is not None:
            self.gauss_kernel_x.requires_grad = False
        if self.gauss_kernel_y is not None:
            self.gauss_kernel_y.requires_grad = False

    def forward(self, x, y):

        self._check_type_forward(x)
        self._check_type_forward(y)
        self._freeze_params()

        if x.shape[1] != y.shape[1]:
            x = torch.mean(x, dim=1, keepdim=True)
            y = torch.mean(y, dim=1, keepdim=True)

        # reshape
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]

        x = x.view(b*c, 1, *spatial_shape)
        y = y.view(b*c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            x = spatial_filter_nd(x, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            y = spatial_filter_nd(y, self.gauss_kernel_y)

        # gradient magnitude
        x_grad_u = torch.abs(spatial_filter_nd(x, self.grad_u_kernel))
        x_grad_v = torch.abs(spatial_filter_nd(x, self.grad_v_kernel))

        y_grad_u = torch.abs(spatial_filter_nd(y, self.grad_u_kernel))
        y_grad_v = torch.abs(spatial_filter_nd(y, self.grad_v_kernel))

        # absolute difference
        diff_u = torch.abs(x_grad_u - y_grad_u)
        diff_v = torch.abs(x_grad_v - y_grad_v)

        # reshape back
        diff_u = diff_u.view(b, c, *spatial_shape)
        diff_v = diff_v.view(b, c, *spatial_shape)

        diff_map = 0.5 * (diff_u + diff_v)

        if self.reduction == 'mean':
            diff = torch.mean(diff_map)
        elif self.reduction == 'sum':
            diff = torch.sum(diff_map)
        else:
            raise KeyError('unsupported reduction type: %s' % self.reduction)

        if self.return_map:
            return diff, diff_map

        return diff


class GradientDifference3d(nn.Module):
    """ Three-dimensional gradient difference

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean'):

        super(GradientDifference3d, self).__init__()

        self.grad_method = grad_method
        self.gauss_sigma = _pair(gauss_sigma)
        self.gauss_truncate = gauss_truncate

        self.grad_u_kernel = None
        self.grad_v_kernel = None
        self.grad_w_kernel = None

        self.gauss_kernel_x = None
        self.gauss_kernel_y = None

        self.return_map = return_map
        self.reduction = reduction

        self._initialize_params()
        self._freeze_params()

    def _initialize_params(self):
        self._initialize_grad_kernel()
        self._initialize_gauss_kernel()

    def _initialize_grad_kernel(self):
        self.grad_u_kernel = _grad_param(3, self.grad_method, axis=0)
        self.grad_v_kernel = _grad_param(3, self.grad_method, axis=1)
        self.grad_w_kernel = _grad_param(3, self.grad_method, axis=2)

    def _initialize_gauss_kernel(self):
        if self.gauss_sigma[0] is not None:
            self.gauss_kernel_x = _gauss_param(3, self.gauss_sigma[0], self.gauss_truncate)
        if self.gauss_sigma[1] is not None:
            self.gauss_kernel_y = _gauss_param(3, self.gauss_sigma[1], self.gauss_truncate)

    def _check_type_forward(self, x):
        if x.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(x.dim()))

    def _freeze_params(self):
        self.grad_u_kernel.requires_grad = False
        self.grad_v_kernel.requires_grad = False
        self.grad_w_kernel.requires_grad = False
        if self.gauss_kernel_x is not None:
            self.gauss_kernel_x.requires_grad = False
        if self.gauss_kernel_y is not None:
            self.gauss_kernel_y.requires_grad = False

    def forward(self, x, y):

        self._check_type_forward(x)
        self._check_type_forward(y)
        self._freeze_params()

        if x.shape[1] != y.shape[1]:
            x = torch.mean(x, dim=1, keepdim=True)
            y = torch.mean(y, dim=1, keepdim=True)

        # reshape
        b, c = x.shape[:2]
        spatial_shape = x.shape[2:]

        x = x.view(b*c, 1, *spatial_shape)
        y = y.view(b*c, 1, *spatial_shape)

        # smoothing
        if self.gauss_kernel_x is not None:
            x = spatial_filter_nd(x, self.gauss_kernel_x)
        if self.gauss_kernel_y is not None:
            y = spatial_filter_nd(y, self.gauss_kernel_y)

        # gradient magnitude
        x_grad_u = torch.abs(spatial_filter_nd(x, self.grad_u_kernel))
        x_grad_v = torch.abs(spatial_filter_nd(x, self.grad_v_kernel))
        x_grad_w = torch.abs(spatial_filter_nd(x, self.grad_w_kernel))

        y_grad_u = torch.abs(spatial_filter_nd(y, self.grad_u_kernel))
        y_grad_v = torch.abs(spatial_filter_nd(y, self.grad_v_kernel))
        y_grad_w = torch.abs(spatial_filter_nd(y, self.grad_w_kernel))

        # absolute difference
        diff_u = torch.abs(x_grad_u - y_grad_u)
        diff_v = torch.abs(x_grad_v - y_grad_v)
        diff_w = torch.abs(x_grad_w - y_grad_w)

        # reshape back
        diff_u = diff_u.view(b, c, *spatial_shape)
        diff_v = diff_v.view(b, c, *spatial_shape)
        diff_w = diff_w.view(b, c, *spatial_shape)

        diff_map = (diff_u + diff_v + diff_w) / 3.0

        if self.reduction == 'mean':
            diff = torch.mean(diff_map)
        elif self.reduction == 'sum':
            diff = torch.sum(diff_map)
        else:
            raise KeyError('unsupported reduction type: %s' % self.reduction)

        if self.return_map:
            return diff, diff_map

        return diff
