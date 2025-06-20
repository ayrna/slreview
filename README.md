# Soft labelling for deep ordinal classification: an experimental review

## Datasets Setup

The datasets directory must be specified in the `data_dir` variable within either `run_experiment.py` or `run_condor.py` (if running with HT-Condor).

The `data_dir` must contain a `datasets.json` file with the following format:

```json
{
    "dataset_name": {
        "train": "relative/path_to/train",
        "test": "relative/path_to/test",
        "type": "image",
        "num_classes": 5
    }
}
```

Each `train` or `test` directory must contain one folder per category in the dataset, following the [torchvision ImageFolder format](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html).


## Running a Single Experiment

You can run a single experiment interactively using:

```bash
python run_experiment.py
```

The experiment settings can be configured in the `config` function inside the `run_experiment.py` file.  
For estimators that require hyperparameters, the first value from each hyperparameter's list, defined in `experiments/utils/estimators.py`, will be used.


## Running Cross-Validation

1. Create a config function in the `run_condor.py` file or use one of the existing ones.  
2. Run with:

    ```bash
    python run_condor.py paramsearch with <config name>
    ```

    For example:

    ```bash
    python run_condor.py paramsearch with victor_config
    ```

## Running Final Training

1. Create a config function in the `run_condor.py` file or use one of the existing ones. This function shares the parameters for both cross-validation and final training. In each case, only the relevant parameters will be used.  
2. Run with:

    ```bash
    python run_condor.py training with <config name>
    ```

    For example:

    ```bash
    python run_condor.py training with victor_config
    ```

# Debugging

To properly debug when using Sacred, you need to add the `-d` option so that Sacred does not shorten the stack trace.
