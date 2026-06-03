"""
Soft IoU loss for semantic segmentation.

Provides a differentiable surrogate of the mean Intersection-over-Union
for pixel-wise classification, designed for the class-imbalanced setting
common in solar granulation segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SoftIoULoss"]


class SoftIoULoss(nn.Module):
    r"""Differentiable soft Intersection-over-Union loss.

    Computes a differentiable approximation of :math:`1 - \mathrm{mIoU}`
    using predicted class probabilities (after softmax) in place of hard
    assignments.

    .. math::
        \mathcal{L} = 1 - \frac{1}{C}\sum_{c=1}^{C}
        \frac{\sum_i \hat{p}_{ic}\, y_{ic} + \epsilon}
             {\sum_i \hat{p}_{ic} + y_{ic}
              - \hat{p}_{ic}\, y_{ic} + \epsilon}

    Parameters
    ----------
    n_classes : int
        Number of semantic classes (default: 2).
    weight : Tensor, optional
        Per-class weights of shape ``(C,)`` (default: ``None``).
    smooth : float, optional
        Smoothing constant to avoid division by zero (default: 1e-6).

    References
    ----------
    Rahman, M. A. & Wang, Y. (2016). Optimizing Intersection-Over-Union
    in Deep Neural Networks for Image Segmentation. *ISVC*, 234–244.
    """

    def __init__(
        self,
        n_classes: int = 2,
        weight: torch.Tensor | None = None,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.smooth = smooth
        if weight is not None:
            self.register_buffer("weight", weight.float())
        else:
            self.weight = None

    @staticmethod
    def _to_one_hot(
        targets: torch.Tensor, n_classes: int
    ) -> torch.Tensor:
        """Convert integer label map to one-hot encoding."""
        N, H, W = targets.shape
        return (
            torch.zeros(N, n_classes, H, W, device=targets.device)
            .scatter_(1, targets.unsqueeze(1), 1.0)
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : Tensor, shape ``(N, C, H, W)``
            Raw logits from the network.
        targets : Tensor, shape ``(N, H, W)``
            Ground-truth class indices in ``[0, C)``.
        """
        N = inputs.size(0)
        probs = F.softmax(inputs, dim=1)
        target_oh = self._to_one_hot(targets, self.n_classes)

        # Intersection and union per class — (N, C)
        inter = (probs * target_oh).view(N, self.n_classes, -1).sum(2)
        union = (probs + target_oh - probs * target_oh).view(
            N, self.n_classes, -1
        ).sum(2)

        iou = (inter + self.smooth) / (union + self.smooth)

        if self.weight is not None:
            iou = (iou * self.weight).sum(dim=1) / self.weight.sum()
        else:
            iou = iou.mean(dim=1)

        return 1.0 - iou.mean()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_classes={self.n_classes}, smooth={self.smooth})"
        )
