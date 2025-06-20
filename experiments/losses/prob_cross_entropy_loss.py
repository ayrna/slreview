from typing import Optional

import torch
from torch import Tensor
from torch.nn import KLDivLoss


class ProbCrossEntropyLoss(torch.nn.Module):
    """CrossEntropyLoss that should receive probabilities as input."""

    def __init__(self, weight: Optional[Tensor] = None) -> None:
        super().__init__()
        self.weight = weight
        self.kld_loss = KLDivLoss(reduction="none")

    def forward(self, input, target) -> Tensor:
        #####################
        input = torch.clamp(input, min=1e-9)
        log_preds = torch.log(input)
        target = torch.clamp(target, min=1e-9)
        #####################
        kld = torch.nn.KLDivLoss(reduction="none")
        kld_loss = kld(log_preds, target)

        if self.weight is not None:
            sample_weight = self.weight[target.argmax(dim=1)]
            kld_loss = kld_loss * sample_weight.unsqueeze(1)

        return kld_loss.sum(dim=1).mean()
