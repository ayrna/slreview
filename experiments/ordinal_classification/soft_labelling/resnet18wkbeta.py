from ...classification import ResNet18Classifier
from ...losses import BetaLoss, WKLossSoftmax


class ResNet18WKBeta(ResNet18Classifier):
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
        loss_eta=0.85,
        params_set="standard",
        penalization_type="quadratic",
    ):
        self.loss_eta = loss_eta
        self.params_set = params_set
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
        )

    def get_loss(self):
        return BetaLoss(
            base_loss=WKLossSoftmax(
                num_classes=self.num_classes,
                weight=self.computed_weights_,
                penalization_type=self.penalization_type,
            ),
            num_classes=self.num_classes,
            params_set=self.params_set,
            eta=self.loss_eta,
        ).to(self.device)
