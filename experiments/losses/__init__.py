from .beta_loss import BetaLoss
from .binomial_loss import BinomialLoss
from .custom_targets_loss import CustomTargetsLoss
from .exponential_loss import ExponentialLoss
from .triangular_loss import TriangularLoss
from .wkloss_softmax import WKLossSoftmax
from .prob_cross_entropy_loss import ProbCrossEntropyLoss

__all__ = [
    "BetaLoss",
    "BinomialLoss",
    "CustomTargetsLoss",
    "TriangularLoss",
    "ExponentialLoss",
    "WKLossSoftmax",
    "ProbCrossEntropyLoss",
]
