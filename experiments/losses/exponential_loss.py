import torch
from dlordinal.soft_labelling import get_exponential_soft_labels
from torch.nn import Module

from .custom_targets_loss import CustomTargetsLoss


class ExponentialLoss(CustomTargetsLoss):
    """Exponential regularised loss from ...

    Parameters
    ----------
    base_loss: Module
        The base loss function.
    num_classes : int
        Number of classes.
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        p: float = 1.0,
        tau: float = 1.0,
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_exponential_soft_labels(num_classes, p, tau))
        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )
