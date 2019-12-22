from __future__ import absolute_import

from .normalized_cross_correlation \
    import NormalizedCrossCorrelation


class NormalizedCrossCorrelationLoss(NormalizedCrossCorrelation):
    """ N-dimensional normalized cross correlation loss (NCC-loss)

    Args:
        eps (float, optional): Epsilon value for numerical stability. Defaults to 1e-8.
        return_map (bool, optional): If True, also return the correlation map. Defaults to False.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``'mean'`` | ``'sum'``. Defaults to ``'mean'``.
    """
    def forward(self, x, y):
        gc = super().forward(x, y)

        if not self.return_map:
            return 1.0 - gc

        return 1.0 - gc[0], gc[1]
