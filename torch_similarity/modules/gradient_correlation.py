from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .gradient_difference import GradientDifference1d
from .gradient_difference import GradientDifference2d
from .gradient_difference import GradientDifference3d
from ..functional import spatial_filter_nd
from ..functional import normalized_cross_correlation


class GradientCorrelation1d(GradientDifference1d):
    """ One-dimensional gradient correlation (GC)

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean',
                 eps=1e-8):

        super().__init__(grad_method,
                        gauss_sigma,
                        gauss_truncate,
                        return_map,
                        reduction)

        self.eps = eps


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

        # gradient correlation
        gc, gc_map = normalized_cross_correlation(x_grad, y_grad, True, self.reduction, self.eps)

        # reshape back
        gc_map = gc_map.view(b, c, *spatial_shape)

        if not self.return_map:
            return gc

        return gc, gc_map


class GradientCorrelation2d(GradientDifference2d):
    """ Two-dimensional gradient correlation (GC)

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean',
                 eps=1e-8):

        super().__init__(grad_method,
                        gauss_sigma,
                        gauss_truncate,
                        return_map,
                        reduction)

        self.eps = eps


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

        # gradient correlation
        gc_u, gc_map_u = normalized_cross_correlation(x_grad_u, y_grad_u, True, self.reduction, self.eps)
        gc_v, gc_map_v = normalized_cross_correlation(x_grad_v, y_grad_v, True, self.reduction, self.eps)

        # reshape back
        gc_map_u = gc_map_u.view(b, c, *spatial_shape)
        gc_map_v = gc_map_v.view(b, c, *spatial_shape)

        gc_map = 0.5 * (gc_map_u + gc_map_v)
        gc = 0.5 * (gc_u + gc_v)

        if not self.return_map:
            return gc

        return gc, gc_map


class GradientCorrelation3d(GradientDifference3d):
    """ Three-dimensional gradient correlation (GC)

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def __init__(self,
                 grad_method='default',
                 gauss_sigma=None,
                 gauss_truncate=4.0,
                 return_map=False,
                 reduction='mean',
                 eps=1e-8):

        super().__init__(grad_method,
                        gauss_sigma,
                        gauss_truncate,
                        return_map,
                        reduction)

        self.eps = eps


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

        # gradient correlation
        gc_u, gc_map_u = normalized_cross_correlation(x_grad_u, y_grad_u, True, self.reduction, self.eps)
        gc_v, gc_map_v = normalized_cross_correlation(x_grad_v, y_grad_v, True, self.reduction, self.eps)
        gc_w, gc_map_w = normalized_cross_correlation(x_grad_w, y_grad_w, True, self.reduction, self.eps)

        # reshape back
        gc_map_u = gc_map_u.view(b, c, *spatial_shape)
        gc_map_v = gc_map_v.view(b, c, *spatial_shape)
        gc_map_w = gc_map_w.view(b, c, *spatial_shape)

        gc_map = (gc_map_u + gc_map_v + gc_map_w) / 3.0
        gc = (gc_u + gc_v + gc_w) / 3.0

        if not self.return_map:
            return gc

        return gc, gc_map
