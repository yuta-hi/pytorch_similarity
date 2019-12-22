from __future__ import absolute_import

from .gradient_correlation import GradientCorrelation1d
from .gradient_correlation import GradientCorrelation2d
from .gradient_correlation import GradientCorrelation3d


class GradientCorrelationLoss1d(GradientCorrelation1d):
    """ One-dimensional gradient correlation loss (GC-loss)

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def forward(self, x, y):
        gc = super().forward(x, y)

        if not self.return_map:
            return 1.0 - gc

        return 1.0 - gc[0], gc[1]


class GradientCorrelationLoss2d(GradientCorrelation2d):
    """ Two-dimensional gradient correlation loss (GC-loss)

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def forward(self, x, y):
        gc = super().forward(x, y)

        if not self.return_map:
            return 1.0 - gc

        return 1.0 - gc[0], gc[1]


class GradientCorrelationLoss3d(GradientCorrelation3d):
    """ Three-dimensional gradient correlation loss (GC-loss)

    Args:
        grad_method (str, optional): Type of the gradient kernel. Defaults to 'default'.
        gauss_sigma (float, optional): Standard deviation for Gaussian kernel. Defaults to None.
        gauss_truncate (float, optional): Truncate the Gaussian kernel at this value. Defaults to 4.0.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
    """
    def forward(self, x, y):
        gc = super().forward(x, y)

        if not self.return_map:
            return 1.0 - gc

        return 1.0 - gc[0], gc[1]
