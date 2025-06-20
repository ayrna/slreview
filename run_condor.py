import json
from hashlib import md5
from pathlib import Path
from shutil import rmtree

import numpy as np
from htcondor import htcondor
from sacred import Experiment

from experiments.utils import (
    CLASSIFIERS,
    ORDINAL_CLASSIFIERS,
    REGRESSORS,
    get_estimator_config,
)
from experiments.utils.paramsearch import get_randomizedsearch_params

ex = Experiment("slreview")


@ex.named_config
def victor_config():
    # estimators = CLASSIFIERS + ORDINAL_CLASSIFIERS + REGRESSORS
    estimators = [
        "resnet18classifier",
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

    # datasets must be registered in data_dir / datasets.json
    datasets = [
        "adience",
        "fgnet",
        "wiki6",
        "utkface12",
        "benelli",
        "alzheimer",
        "ava",
        "retinopathy",
        "smear",
        "hands_color",
        "limuc",
        "knee_kaggle",
        # "tiny_adience",
        # "tiny_utkface12",
        # "tiny_wiki6",
        # "tiny_alzheimer",
        # "tiny_retinopathy",
        # "tiny_ava",
    ]
    data_dir = "/mnt/datasets"
    seeds = 20
    n_folds = 0
    val_size = 0.3
    search_n_iter = 15
    cpus = 1
    n_jobs = 3
    memory = 3072
    gpus = 1
    gpumemory = 8192
    results_dir = "./results"
    tasks_config_dir = "./tasks_config"
    condor_output_dir = "./condor_output"
    priority = 1
    batch_size = 128
    val_metric = "AMAE"
    greater_is_better = False

    # Override memory for specific estimators / datasets
    # Use * as wildcard for all estimators / datasets
    memory_override = {}
    gpumemory_override = {}

    # Override batch_size for specific estimators / datasets
    # Use * as wildcard for all estimators / datasets
    batch_size_override = {}


# python run_condor.py paramsearch with victor_config
@ex.command
def paramsearch(
    estimators,
    datasets,
    data_dir,
    seeds,
    n_folds,
    val_size,
    search_n_iter,
    cpus,
    n_jobs,
    memory,
    gpus,
    gpumemory,
    results_dir,
    tasks_config_dir,
    condor_output_dir,
    priority,
    batch_size,
    memory_override,
    gpumemory_override,
    batch_size_override,
    _config,
):
    print("Running condor param search")
    jobs_data = {}
    for estimator in estimators:
        jobs_data[estimator] = {}
        for dataset in datasets:
            jobs_data[estimator][dataset] = []
            for seed in range(seeds):
                base_config, param_grid = get_estimator_config(estimator)
                param_configs = get_randomizedsearch_params(
                    param_grid, search_n_iter, random_state=seed
                )
                for param_config in param_configs:
                    for fold in range(n_folds if n_folds > 0 else 1):
                        jobs_data[estimator][dataset].append(
                            {
                                "task_config": {
                                    "cpus": cpus,
                                    "memory": _get_value_for_estimator_dataset(
                                        estimator, dataset, memory_override, memory
                                    ),
                                    "gpus": gpus,
                                    "gpumemory": _get_value_for_estimator_dataset(
                                        estimator,
                                        dataset,
                                        gpumemory_override,
                                        gpumemory,
                                    ),
                                    "experiment_config": "",
                                    # The next elements are included just for using them
                                    # in the submit file to determine output file names
                                    "estimator_name": estimator,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "fold": fold,
                                },
                                "experiment_config": {
                                    "data_dir": data_dir,
                                    "estimator_name": estimator,
                                    "dataset": dataset,
                                    "seed": seed,
                                    "n_folds": n_folds if n_folds > 0 else None,
                                    "fold": fold if n_folds > 0 else None,
                                    "val_size": val_size,
                                    "results_dir": results_dir,
                                    "estimator_config": {**base_config, **param_config},
                                    "batch_size": _get_value_for_estimator_dataset(
                                        estimator,
                                        dataset,
                                        batch_size_override,
                                        batch_size,
                                    ),
                                    "interactive": False,
                                    "n_jobs": n_jobs,
                                },
                            }
                        )

    _launch_condor_jobs(
        jobs_data, "paramsearch", tasks_config_dir, condor_output_dir, priority
    )


@ex.command
def training(
    estimators,
    datasets,
    data_dir,
    seeds,
    cpus,
    n_jobs,
    memory,
    gpus,
    gpumemory,
    results_dir,
    tasks_config_dir,
    condor_output_dir,
    priority,
    batch_size,
    val_metric,
    greater_is_better,
    memory_override,
    gpumemory_override,
    batch_size_override,
    _config,
):
    """Runs the final training experiments using the best parameters found for each
    estimator, dataset and random seed. Before running this command, the best
    parameters for each estimator, dataset and random seed must be found using
    the paramsearch command. The best parameters are found by selecting the
    parameters that maximize the validation metric. The best parameters are
    selected from the results directory using the val_metric parameter.
    """

    print("Running condor training")
    estimator_configs = _find_best_estimator_configs(
        results_dir,
        val_metric,
        greater_is_better,
        estimators,
        datasets,
    )

    jobs_data = {}
    for estimator in estimators:
        jobs_data[estimator] = {}
        for dataset in datasets:
            jobs_data[estimator][dataset] = []
            for seed in range(seeds):
                if (estimator, dataset, seed) not in estimator_configs:
                    raise ValueError(
                        f"Best estimator config not found for"
                        f" {estimator}, {dataset}, {seed}"
                    )
                estimator_config = estimator_configs[(estimator, dataset, seed)]
                jobs_data[estimator][dataset].append(
                    {
                        "task_config": {
                            "cpus": cpus,
                            "memory": _get_value_for_estimator_dataset(
                                estimator, dataset, memory_override, memory
                            ),
                            "gpus": gpus,
                            "gpumemory": _get_value_for_estimator_dataset(
                                estimator, dataset, gpumemory_override, gpumemory
                            ),
                            "experiment_config": "",
                            # The next elements are included just for using them
                            # in the submit file to determine output file names
                            "estimator_name": estimator,
                            "dataset": dataset,
                            "seed": seed,
                            "fold": "final",
                        },
                        "experiment_config": {
                            "data_dir": data_dir,
                            "estimator_name": estimator,
                            "dataset": dataset,
                            "seed": seed,
                            "n_folds": None,
                            "fold": None,
                            "val_size": None,
                            "results_dir": results_dir,
                            "estimator_config": estimator_config,
                            "interactive": False,
                            "batch_size": _get_value_for_estimator_dataset(
                                estimator, dataset, batch_size_override, batch_size
                            ),
                            "n_jobs": n_jobs,
                        },
                    }
                )

    _launch_condor_jobs(
        jobs_data, "training", tasks_config_dir, condor_output_dir, priority
    )


def _launch_condor_jobs(jobs_data, name, tasks_config_dir, condor_output_dir, priority):
    tasks_config_dir = Path(tasks_config_dir)
    rmtree(tasks_config_dir, ignore_errors=True)

    total_jobs_count = sum(
        sum(len(jobs) for jobs in datasets.values()) for datasets in jobs_data.values()
    )
    job_count = 0

    print("Creating experiment config json files: ")
    for estimator, datasets in jobs_data.items():
        for dataset, jobs in datasets.items():
            # Create a json config directory for each estimator and dataset
            est_dat_dir = tasks_config_dir / estimator / dataset
            est_dat_dir.mkdir(parents=True, exist_ok=True)

            for job_data in jobs:
                job_count += 1
                # Compute de hash of the experiment config to identify each config
                job_hash = job_data["task_config"]["job_hash"] = md5(
                    json.dumps(job_data["experiment_config"], sort_keys=True).encode(
                        "utf-8"
                    )
                ).hexdigest()

                # Save the path of the json file in the task_config dictionary
                job_data["task_config"]["experiment_config"] = str(
                    est_dat_dir / f"s{job_data['task_config']['seed']}"
                    f"_f{job_data['task_config']['fold']}_{job_hash}.json"
                )

                # Save the experiment config into a json file
                with open(job_data["task_config"]["experiment_config"], "w") as f:
                    json.dump(job_data["experiment_config"], f)

                # Convert all items from task_config to string to avoid error with condor
                for key, value in job_data["task_config"].items():
                    job_data["task_config"][key] = str(value)

                print(f"{job_count}/{total_jobs_count}", end="\r")

    print("")

    condor_output_dir = Path(condor_output_dir)
    rmtree(condor_output_dir, ignore_errors=True)
    condor_output_dir.mkdir(parents=True, exist_ok=True)

    for estimator, datasets in jobs_data.items():
        for dataset, jobs in datasets.items():
            # Create condor output directories
            (condor_output_dir / estimator / dataset).mkdir(parents=True, exist_ok=True)

            schedd = htcondor.Schedd()
            job = htcondor.Submit(
                {
                    "executable": "run_experiment.sh",
                    "arguments": "with $(experiment_config)",
                    "getenv": "True",
                    "output": f"{str(condor_output_dir)}/$(estimator_name)/$(dataset)/output_s$(seed)_f$(fold)_$(job_hash).out",
                    "error": f"{str(condor_output_dir)}/$(estimator_name)/$(dataset)/error_s$(seed)_f$(fold)_$(job_hash).err",
                    "log": f"{str(condor_output_dir)}/$(estimator_name)/$(dataset)/log_s$(seed)_f$(fold)_$(job_hash).log",
                    "should_transfer_files": "NO",
                    "request_GPUs": "$(gpus)",
                    "require_GPUs": "GlobalMemoryMb >= $(gpumemory)",
                    "request_CPUs": "$(cpus)",
                    "request_memory": "$(memory)",
                    "batch_name": f"{estimator}_{dataset}_{name}",
                    "priority": priority,
                    "on_exit_hold": "ExitBySignal == True || ExitCode != 0",
                }
            )

            submit_result = schedd.submit(
                job, itemdata=iter(map(lambda j: j["task_config"], jobs))
            )
            print(
                f"{submit_result.num_procs()} jobs submitted"
                f" as cluster {submit_result.cluster()}"
                f" for {estimator} and {dataset}."
            )


def _find_best_estimator_configs(
    results_dir, val_metric, greater_is_better, estimators, datasets
):
    from remayn.result_set import ResultFolder

    from experiments.utils import compute_metrics

    def filter_fn(result):
        if not result.config["estimator_name"] in estimators:
            return False

        if not result.config["dataset"] in datasets:
            return False

        if (result.config["n_folds"] is None or result.config["n_folds"] <= 1) and (
            result.config["val_size"] is None or result.config["val_size"] <= 0
        ):
            return False

        return True

    results = ResultFolder(results_dir)
    print(results)

    df = results.create_dataframe(
        config_columns=[
            "estimator_name",
            "dataset",
            "rs",
            "estimator_config",
        ],
        best_params_columns=["max_iter", "thresholds_max_iter", "final_max_iter"],
        metrics_fn=compute_metrics,
        filter_fn=filter_fn,
        include_train=False,
        include_val=True,
        config_columns_prefix="",
    )

    if greater_is_better:
        grouped_df = df.loc[
            df.groupby(["estimator_name", "dataset", "rs"])[
                f"val_{val_metric}"
            ].idxmax()
        ]
    else:
        grouped_df = df.loc[
            df.groupby(["estimator_name", "dataset", "rs"])[
                f"val_{val_metric}"
            ].idxmin()
        ]

    configs = {}
    for idx, row in grouped_df.iterrows():
        estimator_name = row["estimator_name"]
        dataset = row["dataset"]
        rs = row["rs"]
        estimator_config = row["estimator_config"]

        # Replace max_iter with the number of epochs of early stopping
        if (
            "best_max_iter" in row
            and not row["best_max_iter"] is None
            and not np.isnan(row["best_max_iter"])
        ):
            estimator_config["max_iter"] = int(row["best_max_iter"])

        configs[(estimator_name, dataset, rs)] = estimator_config

    return configs


def _get_value_for_estimator_dataset(estimator, dataset, override, value):
    if override is not None:
        if estimator in override and dataset in override[estimator]:
            return override[estimator][dataset]
        elif "*" in override and dataset in override["*"]:
            return override["*"][dataset]
        elif estimator in override and "*" in override[estimator]:
            return override[estimator]["*"]
    return value


if __name__ == "__main__":
    ex.run_commandline()
