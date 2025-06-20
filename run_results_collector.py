import argparse

parser = argparse.ArgumentParser(description="Run results collector")
parser.add_argument("appendix", type=str, help="Appendix for the output files")
parser.add_argument("--skip-zip", action="store_true", help="Skip zipping the results")
args = parser.parse_args()
appendix = args.appendix

from datetime import datetime
from pathlib import Path

import pandas as pd
from remayn.report import create_excel_columns_report, create_excel_summary_report
from remayn.result_set import ResultFolder

from experiments.utils import compute_metrics

path = Path("./results")
results = ResultFolder(path)
print("Results loaded")

# Fields from the experiment config that will be included in the dataframe as columns
config_columns_to_include = [
    "dataset",
    "estimator_name",
    "rs",
    "estimator_config.learning_rate",
    "estimator_config.loss_eta",
    "estimator_config.loss_p",
    "estimator_config.loss_alpha2",
    "estimator_config.max_iter",
    "estimator_config.link_function",
    "estimator_config.penalization_type",
]
best_params_columns_to_include = []
filter_methods = [
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
filter_datasets = [
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
]

seeds = list(range(20))


def filter_fn(result):
    if (
        len(filter_methods) > 0
        and result.config["estimator_name"] not in filter_methods
    ):
        return False

    if len(filter_datasets) > 0 and result.config["dataset"] not in filter_datasets:
        return False

    if result.config["rs"] not in seeds:
        return False

    # Get only final results
    if result.config["n_folds"] is not None and result.config["n_folds"] > 1:
        return False
    if result.config["val_size"] is not None and result.config["val_size"] > 0:
        return False

    return True


from joblib import parallel_backend

# with parallel_backend("multiprocessing"):
df = results.create_dataframe(
    config_columns=config_columns_to_include,
    best_params_columns=best_params_columns_to_include,
    filter_fn=filter_fn,
    metrics_fn=compute_metrics,
    include_train=True,
    include_val=False,
    config_columns_prefix="",
)

df.sort_values(by=["dataset", "estimator_name", "rs"], inplace=True)

group_columns = ["dataset", "estimator_name"]

print(df)

output_path_wo_ext = (
    f'prepared_results/{datetime.now().strftime(r"%Y%m%d_%H%M%S")}_{appendix}'
)

metrics_example = compute_metrics([1], [[1.0, 2.0]])
metric_columns = list(metrics_example.keys())

Path(output_path_wo_ext).parent.mkdir(parents=True, exist_ok=True)
df.to_excel(f"{output_path_wo_ext}.xlsx", index=False)

# with pd.ExcelWriter(f"{output_path_wo_ext}.xlsx", mode="w") as writer:
#     create_excel_summary_report(
#         df, f"{output_path_wo_ext}.xlsx", group_columns, excel_writer=writer
#     )
#     create_excel_columns_report(
#         df,
#         f"{output_path_wo_ext}.xlsx",
#         metric_columns=metric_columns,
#         pivot_index="rs",
#         pivot_columns=["estimator_name", "dataset"],
#         excel_writer=writer,
#     )

if not args.skip_zip:
    from shutil import make_archive

    make_archive(output_path_wo_ext, "zip", str(path))
