# -*- coding: utf-8 -*-
import json
import time

import numpy as np


def load_and_run_experiment(
    *,
    data_dir,
    dataset,
    n_folds=None,
    val_size=None,
    fold=None,
    results_dir="./results",
    estimator_name="resnet18classifier",
    estimator_config={},
    batch_size=128,
    seed=0,
    interactive=False,
    n_jobs=1,
):
    if n_folds is not None and n_folds > 1 and val_size is not None and val_size > 0:
        raise ValueError("Only one of n_folds and val_size can be set")

    from os import environ
    from random import seed as random_seed

    from remayn.result import make_result
    from remayn.result_set import ResultFolder
    from torch import cuda, manual_seed, use_deterministic_algorithms

    from . import (
        create_dataset_resample,
        get_dataset_fold,
        get_dataset_holdout,
        get_estimator,
        load_data,
    )

    # Fix seeds
    np.random.seed(seed)
    manual_seed(seed)
    random_seed(seed)
    environ["PYTHONHASHSEED"] = str(seed)
    environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    use_deterministic_algorithms(True)

    train_dataset = load_data(data_dir=data_dir, dataset=dataset, partition="train")
    val_dataset = None
    test_dataset = load_data(data_dir=data_dir, dataset=dataset, partition="test")

    if seed != 0:
        train_dataset, test_dataset = create_dataset_resample(
            train_dataset, test_dataset, random_state=seed
        )

    print(
        f"Loaded {dataset} dataset with {len(train_dataset)} train samples and "
        f"{len(test_dataset)} test samples."
    )

    # Can be used to check the resampling
    # print("Train samples per class:")
    # train_samples_per_class = np.unique(train_dataset.targets, return_counts=True)[1]
    # print(train_samples_per_class / np.sum(train_samples_per_class))
    # print(train_dataset.targets)
    # print("Test samples per class:")
    # test_samples_per_class = np.unique(test_dataset.targets, return_counts=True)[1]
    # print(test_samples_per_class / np.sum(test_samples_per_class))
    # print(test_dataset.targets)

    if hasattr(train_dataset, "classes"):
        num_classes = len(train_dataset.classes)
    else:
        raise ValueError("Number of classes could not be inferred from dataset")

    if n_folds is not None and n_folds > 1:
        train_dataset, val_dataset = get_dataset_fold(
            train_dataset, fold=fold, n_folds=n_folds, random_state=seed
        )
        print(
            f"Using {n_folds}-fold cross-validation with fold {fold}.\n"
            f"Train samples: {len(train_dataset)}, val samples: {len(val_dataset)}"
        )

    if val_size is not None and val_size > 0:
        train_dataset, val_dataset = get_dataset_holdout(
            train_dataset, test_size=val_size, random_state=seed
        )
        print(
            f"Using holdout validation with val_size {val_size}.\n"
            f"Train samples: {len(train_dataset)}, val samples: {len(val_dataset)}"
        )

    # if estimator_config is not set, use the first gridsearch config
    if estimator_config is None:
        from experiments.utils import get_estimator_config, get_gridsearch_params

        base_config, param_grid = get_estimator_config(estimator_name)
        estimator_configs = get_gridsearch_params(param_grid)
        estimator_config = {**base_config, **estimator_configs[0]}

    # Shuffled targets. Passed to get_estimator for computing class weights
    train_targets = train_dataset.targets

    estimator = get_estimator(
        estimator_name,
        estimator_config,
        num_classes=num_classes,
        train_targets=train_targets,
        val_dataset=val_dataset,  # slows down training process
        random_state=seed,
        n_jobs=n_jobs,
        batch_size=batch_size,
    )

    experiment_config = get_experiment_config(
        estimator, estimator_name, dataset, seed, 0, n_folds, fold, val_size
    )

    if not interactive:
        results = ResultFolder(results_dir)
        if experiment_config in results:
            print("Experiment already run")
            return

    print("Running experiment with config (including estimator config):")
    print(json.dumps(experiment_config, indent=4))

    if cuda.is_available():
        print("GPU available")

    start = int(round(time.time() * 1000))
    estimator.fit(train_dataset)

    train_probs = estimator.predict_proba(train_dataset)
    test_probs = estimator.predict_proba(test_dataset)

    if val_dataset is not None:
        val_probs = estimator.predict_proba(val_dataset)
        val_targets = val_dataset.targets
    else:
        val_probs = None
        val_targets = None

    train_targets = train_dataset.targets
    test_targets = test_dataset.targets

    total_time = int(round(time.time() * 1000)) - start

    experiment_config = get_experiment_config(
        estimator,
        estimator_name,
        dataset,
        seed,
        0,
        n_folds,
        fold,
        val_size,
    )

    train_history = (
        estimator.train_history if hasattr(estimator, "train_history") else None
    )
    val_history = (
        estimator.valid_history if hasattr(estimator, "valid_history") else None
    )

    if not interactive:
        result = make_result(
            base_path=results_dir,
            config=experiment_config,
            predictions=np.array(test_probs),
            targets=np.array(test_targets),
            train_predictions=np.array(train_probs),
            train_targets=np.array(train_targets),
            val_predictions=np.array(val_probs),
            val_targets=np.array(val_targets),
            time=total_time,
            best_params=estimator.best_params_,
            best_model=None,
            train_history=np.array(train_history),
            val_history=np.array(val_history),
        )
        result.save()
    else:
        if hasattr(estimator, "best_params_"):
            print("best_params")
            print(json.dumps(estimator.best_params_, indent=4))

        train_metrics = compute_metrics(train_targets, train_probs)
        print("train_metrics")
        print(json.dumps(train_metrics, indent=4))

        if val_probs is not None:
            val_metrics = compute_metrics(val_targets, val_probs)
            print("val_metrics")
            print(json.dumps(val_metrics, indent=4))

        test_metrics = compute_metrics(test_targets, test_probs)
        print("test_metrics")
        print(json.dumps(test_metrics, indent=4))

        if train_history is not None:
            print("train_history")
            print(train_history)

        if val_history is not None:
            print("val_history")
            print(val_history)


def get_experiment_config(
    estimator, estimator_name, dataset, rs, resample_id, n_folds, fold, val_size
):
    config = {}
    config["estimator_config"] = estimator.get_params().copy()
    config["estimator_name"] = estimator_name
    config["dataset"] = dataset
    config["rs"] = rs
    config["resample_id"] = resample_id
    config["n_folds"] = n_folds
    config["fold"] = fold
    config["val_size"] = val_size

    estimator_params_to_remove = [
        "num_classes",
        "train_targets",
        "val_dataset",
        "verbose",
        "n_jobs",
        "batch_size",
    ]

    for param in estimator_params_to_remove:
        if param in config["estimator_config"]:
            del config["estimator_config"][param]

    return config


def compute_metrics(targets, probabilities):
    from dlmisc.metrics import accuracy_off1, minimum_sensitivity
    from scipy.special import softmax
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        mean_absolute_error,
        recall_score,
    )

    from experiments.metrics import amae, mmae, rank_probability_score, mes

    targets = np.array(targets)
    probabilities = np.array(probabilities)

    if len(probabilities.shape) > 1:
        predictions = np.argmax(probabilities, axis=1)
    else:
        predictions = np.array(probabilities)

    metrics = {
        "QWK": cohen_kappa_score(targets, predictions, weights="quadratic"),
        "MAE": mean_absolute_error(targets, predictions),
        "1-off": accuracy_off1(targets, predictions),
        "CCR": accuracy_score(targets, predictions),
        "MS": minimum_sensitivity(targets, predictions),
        "BalancedAccuracy": balanced_accuracy_score(targets, predictions),
        "AMAE": amae(targets, predictions),
        "MMAE": mmae(targets, predictions),
        "RPS": rank_probability_score(targets, softmax(probabilities, axis=1)),
        "MES": mes(targets, predictions),
    }

    # Compute sensitivities for each class
    sensitivities = np.array(recall_score(targets, predictions, average=None))

    for i, sens in enumerate(sensitivities):
        metrics[f"Sens{i}"] = sens

    return metrics
