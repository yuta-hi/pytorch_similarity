from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.nn.functional as F

_func_conv_nd_table = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d
}

def spatial_filter_nd(x, kernel, mode='replicate'):
    """ N-dimensional spatial filter with padding.

    Args:
        x (~torch.Tensor): Input tensor.
        kernel (~torch.Tensor): Weight tensor (e.g., Gaussain kernel).
        mode (str, optional): Padding mode. Defaults to 'replicate'.

    Returns:
        ~torch.Tensor: Output tensor
    """

    n_dim = x.dim() - 2
    conv = _func_conv_nd_table[n_dim]

    pad = [None,None]*n_dim
    pad[0::2] = kernel.shape[2:]
    pad[1::2] = kernel.shape[2:]
    pad = [k//2 for k in pad]

    return conv(F.pad(x, pad=pad, mode=mode), kernel)

def normalized_cross_correlation(x, y, return_map, reduction='mean', eps=1e-8):
    """ N-dimensional normalized cross correlation (NCC)

    Args:
        x (~torch.Tensor): Input tensor.
        y (~torch.Tensor): Input tensor.
        return_map (bool): If True, also return the correlation map.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'sum'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.

    Returns:
        ~torch.Tensor: Output scalar
        ~torch.Tensor: Output tensor
    """

    shape = x.shape
    b = shape[0]

    # reshape
    x = x.view(b, -1)
    y = y.view(b, -1)

    # mean
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # deviation
    x = x - x_mean
    y = y - y_mean

    dev_xy = torch.mul(x,y)
    dev_xx = torch.mul(x,x)
    dev_yy = torch.mul(y,y)

    dev_xx_sum = torch.sum(dev_xx, dim=1, keepdim=True)
    dev_yy_sum = torch.sum(dev_yy, dim=1, keepdim=True)

    ncc = torch.div(dev_xy + eps / dev_xy.shape[1],
                    torch.sqrt( torch.mul(dev_xx_sum, dev_yy_sum)) + eps)
    ncc_map = ncc.view(b, *shape[1:])

    # reduce
    if reduction == 'mean':
        ncc = torch.mean(torch.sum(ncc, dim=1))
    elif reduction == 'sum':
        ncc = torch.sum(ncc)
    else:
        raise KeyError('unsupported reduction type: %s' % reduction)

    if not return_map:
        return ncc

    return ncc, ncc_map


def local_contrast_norm_nd(x, kernel, eps=1e-8):
    """ N-dimensional local contrast normalization (LCN).

    Args:
        x (~torch.Tensor): Input tensor.
        kernel (~torch.Tensor): Weight tensor (e.g., Gaussain kernel).
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.

    Returns:
        ~torch.Tensor: Output tensor
    """

    # reshape
    b, c = x.shape[:2]
    spatial_shape = x.shape[2:]

    x = x.view(b*c, 1, *spatial_shape)

    # local mean
    x_mean = spatial_filter_nd(x, kernel)

    # subtractive normalization
    x_sub = x - x_mean

    # local deviation
    x_dev = spatial_filter_nd(x_sub.pow(2), kernel)
    x_dev = x_dev.sqrt()
    x_dev_mean = x_dev.mean()

    # divisive normalization
    x_dev = torch.max(x_dev_mean, x_dev)
    x_dev = torch.clamp(x_dev, eps)

    ret = x_sub / x_dev
    ret = ret.view(b, c, *spatial_shape)

    return ret
