import numpy as np

REGRESSORS = []
CLASSIFIERS = ["resnet18classifier"]
ORDINAL_CLASSIFIERS = [
    "resnet18poissonclassifier",
    "resnet18binomialclassifier",
    "resnet18exponentialclassifier",
    "resnet18betaclassifier",
    "resnet18triangularclassifier",
    "resnet18clmclassifier",
    "resnet18clmwkclassifier",
    "resnet18wkclassifier",
    "resnet18wkbetaclassifier",
    "resnet18wkbinomialclassifier",
    "resnet18wkexponentialclassifier",
    "resnet18wktriangularclassifier",
    "resnet18clmbetaclassifier",
    "resnet18clmbinomialclassifier",
    "resnet18clmexponentialclassifier",
    "resnet18clmtriangularclassifier",
    "resnet18clmwkbetaclassifier",
    "resnet18clmwkbinomialclassifier",
    "resnet18clmwkexponentialclassifier",
    "resnet18clmwktriangularclassifier",
]


def get_estimator_config(estimator_name):
    if estimator_name in REGRESSORS:
        raise ValueError(f"Estimator {estimator_name} not found")
    elif estimator_name in CLASSIFIERS:
        if estimator_name == "resnet18classifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
            }

            return base_config, param_grid
        else:
            raise ValueError(f"Estimator {estimator_name} not found")

    elif estimator_name in ORDINAL_CLASSIFIERS:
        if estimator_name == "resnet18poissonclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "loss_eta": [0.8, 1.0],
            }

            return base_config, param_grid
        elif estimator_name == "resnet18binomialclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "loss_eta": [0.8, 1.0],
            }

            return base_config, param_grid
        elif estimator_name == "resnet18exponentialclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "loss_p": [1.0, 1.5, 2.0],
                "loss_eta": [0.8, 1.0],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18betaclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "loss_eta": [0.8, 1.0],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18triangularclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "loss_alpha2": [0.01, 0.05, 0.10],
                "loss_eta": [0.8, 1.0],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 48,
            }

            param_grid = {
                "learning_rate": [1e-2, 1e-3, 1e-4],
                "link_function": ["logit"],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmwkclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
                "min_distance": 0.5,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "link_function": ["logit"],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18wkclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18wkbinomialclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "loss_eta": [0.8, 1.0],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid
        elif estimator_name == "resnet18wkexponentialclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "loss_p": [1.0, 1.5, 2.0],
                "loss_eta": [0.8, 1.0],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18wkbetaclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "loss_eta": [0.8, 1.0],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18wktriangularclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "loss_alpha2": [0.01, 0.05, 0.10],
                "loss_eta": [0.8, 1.0],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmbetaclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "link_function": ["logit"],
                "loss_eta": [0.8, 1.0],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmexponentialclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "link_function": ["logit"],
                "loss_p": [1.0, 1.5, 2.0],
                "loss_eta": [0.8, 1.0],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmbinomialclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "link_function": ["logit"],
                "loss_eta": [0.8, 1.0],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmtriangularclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "link_function": ["logit"],
                "loss_alpha2": [0.01, 0.05, 0.10],
                "loss_eta": [0.8, 1.0],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmwkbetaclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "link_function": ["logit"],
                "loss_eta": [0.8, 1.0],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmwkexponentialclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "link_function": ["logit"],
                "loss_p": [1.0, 1.5, 2.0],
                "loss_eta": [0.8, 1.0],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmwkbinomialclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "link_function": ["logit"],
                "loss_eta": [0.8, 1.0],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid

        elif estimator_name == "resnet18clmwktriangularclassifier":
            base_config = {
                "class_weight": "balanced",
                "verbose": 3,
                "max_iter": 100,
            }

            param_grid = {
                "learning_rate": [1e-3, 1e-2, 1e-4],
                "link_function": ["logit"],
                "loss_alpha2": [0.01, 0.05, 0.10],
                "loss_eta": [0.8, 1.0],
                "penalization_type": ["quadratic", "linear"],
            }

            return base_config, param_grid

        raise ValueError(f"Estimator {estimator_name} not found")
    else:
        raise ValueError(f"Estimator {estimator_name} not found")


def get_estimator(
    estimator_name,
    config,
    *,
    num_classes,
    train_targets,
    val_dataset,
    random_state=0,
    n_jobs=1,
    batch_size=128,
):
    from copy import copy

    config = copy(config)
    if "device" in config:
        del config["device"]

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if estimator_name in REGRESSORS:
        raise ValueError(f"Estimator {estimator_name} not found")
    elif estimator_name in CLASSIFIERS:
        if estimator_name == "resnet18classifier":
            from ..classification import ResNet18Classifier

            estimator = ResNet18Classifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )
        else:
            raise ValueError(f"Estimator {estimator_name} not found")

    elif estimator_name in ORDINAL_CLASSIFIERS:
        if estimator_name == "resnet18poissonclassifier":
            from ..ordinal_classification.soft_labelling import ResNet18Poisson

            estimator = ResNet18Poisson(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )
        elif estimator_name == "resnet18binomialclassifier":
            from ..ordinal_classification.soft_labelling import ResNet18Binomial

            estimator = ResNet18Binomial(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )
        elif estimator_name == "resnet18exponentialclassifier":
            from ..ordinal_classification.soft_labelling import ResNet18Exponential

            estimator = ResNet18Exponential(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18betaclassifier":
            from ..ordinal_classification.soft_labelling import ResNet18Beta

            estimator = ResNet18Beta(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18triangularclassifier":
            from ..ordinal_classification.soft_labelling import ResNet18Triangular

            estimator = ResNet18Triangular(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmclassifier":
            from ..ordinal_classification.thresholds import ResNet18CLMClassifier

            estimator = ResNet18CLMClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmwkclassifier":
            from ..ordinal_classification.thresholds import ResNet18CLMWKClassifier

            estimator = ResNet18CLMWKClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18wkclassifier":
            from ..ordinal_classification import ResNet18WKClassifier

            estimator = ResNet18WKClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18wkbetaclassifier":
            from ..ordinal_classification.soft_labelling import ResNet18WKBeta

            estimator = ResNet18WKBeta(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18wkbinomialclassifier":
            from ..ordinal_classification.soft_labelling import ResNet18WKBinomial

            estimator = ResNet18WKBinomial(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18wkexponentialclassifier":
            from ..ordinal_classification.soft_labelling import ResNet18WKExponential

            estimator = ResNet18WKExponential(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18wktriangularclassifier":
            from ..ordinal_classification.soft_labelling import ResNet18WKTriangular

            estimator = ResNet18WKTriangular(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmbetaclassifier":
            from ..ordinal_classification.thresholds import ResNet18CLMBetaClassifier

            estimator = ResNet18CLMBetaClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmexponentialclassifier":
            from ..ordinal_classification.thresholds import (
                ResNet18CLMExponentialClassifier,
            )

            estimator = ResNet18CLMExponentialClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmbinomialclassifier":
            from ..ordinal_classification.thresholds import (
                ResNet18CLMBinomialClassifier,
            )

            estimator = ResNet18CLMBinomialClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmtriangularclassifier":
            from ..ordinal_classification.thresholds import (
                ResNet18CLMTriangularClassifier,
            )

            estimator = ResNet18CLMTriangularClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmwkbetaclassifier":
            from ..ordinal_classification.thresholds import ResNet18CLMWKBetaClassifier

            estimator = ResNet18CLMWKBetaClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmwkbinomialclassifier":
            from ..ordinal_classification.thresholds import (
                ResNet18CLMWKBinomialClassifier,
            )

            estimator = ResNet18CLMWKBinomialClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmwkexponentialclassifier":
            from ..ordinal_classification.thresholds import (
                ResNet18CLMWKExponentialClassifier,
            )

            estimator = ResNet18CLMWKExponentialClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        elif estimator_name == "resnet18clmwktriangularclassifier":
            from ..ordinal_classification.thresholds import (
                ResNet18CLMWKTriangularClassifier,
            )

            estimator = ResNet18CLMWKTriangularClassifier(
                **config,
                num_classes=num_classes,
                train_targets=train_targets,
                val_dataset=val_dataset,
                n_jobs=n_jobs,
                batch_size=batch_size,
                device=device,
            )

        else:
            raise ValueError(f"Estimator {estimator_name} not found")
    else:
        raise ValueError(f"Estimator {estimator_name} not found")

    return estimator
