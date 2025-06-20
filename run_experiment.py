from sacred import Experiment

ex = Experiment("slreview")


@ex.config
def config():
    data_dir = "/mnt/datasets"
    dataset = "fgnet"
    n_folds = None
    fold = None
    val_size = 0.0  # 0.3
    results_dir = "./results"
    estimator_name = "resnet18classifier"
    estimator_config = None
    batch_size = 128
    seed = 0
    interactive = True
    n_jobs = 1


@ex.main
def main(
    data_dir,
    dataset,
    n_folds,
    fold,
    val_size,
    results_dir,
    estimator_name,
    estimator_config,
    batch_size,
    seed,
    interactive,
    n_jobs,
    _config,
):
    import json

    from experiments.utils import load_and_run_experiment

    print(f"Running experiment with config: ")
    print(json.dumps(_config, indent=4))

    load_and_run_experiment(
        data_dir=data_dir,
        dataset=dataset,
        n_folds=n_folds,
        fold=fold,
        val_size=val_size,
        results_dir=results_dir,
        estimator_name=estimator_name,
        estimator_config=estimator_config,
        batch_size=batch_size,
        seed=seed,
        interactive=interactive,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    ex.run_commandline()
