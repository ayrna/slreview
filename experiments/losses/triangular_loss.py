import torch
from dlordinal.soft_labelling import get_triangular_soft_labels
from torch.nn import Module

from .custom_targets_loss import CustomTargetsLoss


class TriangularLoss(CustomTargetsLoss):
    """Triangular regularised loss from :footcite:t:`vargas2023softlabelling`.

    Parameters
    ----------
    base_loss: Module
        The base loss function.
    num_classes : int
        Number of classes.
    alpha2 : float, default=0.05
        Parameter that controls the probability deposited in adjacent classes.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        alpha2: float = 0.05,
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_triangular_soft_labels(num_classes, alpha2))
        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )
