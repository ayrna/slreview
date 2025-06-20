from dlordinal.losses import WKLoss

from ...losses import BinomialLoss
from .resnet18clmclassifier import ResNet18CLMClassifier


class ResNet18CLMWKBinomialClassifier(ResNet18CLMClassifier):
    def __init__(
        self,
        *,
        num_classes,
        device="cpu",
        max_iter=1000,
        class_weight=None,
        train_targets=None,
        learning_rate=1e-3,
        verbose=0,
        batch_size=128,
        val_dataset=None,
        n_jobs=1,
        link_function="logit",
        min_distance=0.0,
        loss_eta=1.0,
        penalization_type="quadratic",
    ):

        self.loss_eta = loss_eta
        self.penalization_type = penalization_type

        super().__init__(
            num_classes=num_classes,
            device=device,
            max_iter=max_iter,
            class_weight=class_weight,
            train_targets=train_targets,
            learning_rate=learning_rate,
            verbose=verbose,
            batch_size=batch_size,
            val_dataset=val_dataset,
            n_jobs=n_jobs,
            link_function=link_function,
            min_distance=min_distance,
        )

    def get_loss(self):
        return BinomialLoss(
            base_loss=WKLoss(
                num_classes=self.num_classes,
                penalization_type=self.penalization_type,
                weight=self.computed_weights_,
            ),
            num_classes=self.num_classes,
            eta=self.loss_eta,
        ).to(self.device)
