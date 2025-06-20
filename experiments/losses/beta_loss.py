import torch
from dlordinal.soft_labelling import get_beta_soft_labels
from torch.nn import Module

from .custom_targets_loss import CustomTargetsLoss


class BetaLoss(CustomTargetsLoss):
    """Beta regularised loss from :footcite:t:`vargas2022unimodal`.

    Parameters
    ----------
    base_loss: Module
        The base loss function.
    num_classes : int
        Number of classes.
    params_set : str, default='standard'
        The set of parameters to use for the beta distribution (chosen from the
        _beta_params_set dictionary).
    eta : float, default=1.0
        Parameter that controls the influence of the regularisation.
    """

    def __init__(
        self,
        base_loss: Module,
        num_classes: int,
        params_set: str = "standard",
        eta: float = 1.0,
    ):
        # Precompute class probabilities for each label
        cls_probs = torch.tensor(get_beta_soft_labels(num_classes, params_set)).float()
        super().__init__(
            base_loss=base_loss,
            cls_probs=cls_probs,
            eta=eta,
        )
