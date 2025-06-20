import json
import warnings
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, Dataset, Subset
from torchvision.datasets import ImageFolder


def get_dataset_info(data_dir, dataset_name):
    data_dir = Path(data_dir)
    with open(data_dir / "datasets.json", "r") as f:
        datasets_info = json.load(f)

    if dataset_name not in datasets_info:
        raise FileNotFoundError(f"Dataset {dataset_name} not found in datasets.json")

    dataset_info = datasets_info[dataset_name]

    if "train" in dataset_info:
        dataset_info["train"] = data_dir / dataset_info["train"]

    if "val" in dataset_info:
        dataset_info["val"] = data_dir / dataset_info["val"]

    if "test" in dataset_info:
        dataset_info["test"] = data_dir / dataset_info["test"]

    return dataset_info


def load_data(*, data_dir, dataset, partition) -> ImageFolder:
    if partition not in ["train", "test"]:
        raise ValueError(f"Partition {partition} not in ['train', 'test']")

    try:
        dataset_info = get_dataset_info(data_dir, dataset)
        data_path = dataset_info[partition]
        data_type = dataset_info["type"]

        if data_type == "image":
            from torch import float32 as torch_float32
            from torchvision import disable_beta_transforms_warning
            from torchvision.datasets import ImageFolder
            from torchvision.transforms import v2 as transforms

            disable_beta_transforms_warning()

            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.ConvertImageDtype(),
                ]
            )

            dataset = ImageFolder(data_path, transform=transform)

            print(f"Loaded {partition} data from {data_path}.")
            return dataset
        else:
            raise NotImplementedError(f"Data type {data_type} not implemented")

    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset {dataset} not found in datasets.json")


def create_dataset_resample(
    train_dataset: Dataset, test_dataset: Dataset, random_state=0
):
    if random_state == 0:
        return train_dataset, test_dataset

    dataset = MyConcatDataset([train_dataset, test_dataset])
    test_size = len(test_dataset)

    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    sss_splits = list(sss.split(X=np.zeros(len(dataset)), y=dataset.targets))
    train_idx, test_idx = sss_splits[0]

    new_train_dataset = MySubset(dataset, train_idx)
    new_test_dataset = MySubset(dataset, test_idx)

    return new_train_dataset, new_test_dataset


class MySubset(Subset):
    """
    Subset of a dataset at specified indices. Includes targets if they are
    available in the original dataset.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        if hasattr(dataset, "targets") and isinstance(dataset.targets, list):  # type: ignore
            self.targets = [dataset.targets[i] for i in indices]  # type: ignore

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.dataset[[self.indices[i] for i in idx]]
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    @property
    def classes(self):
        return np.unique(self.targets)


class MyConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super(MyConcatDataset, self).__init__(datasets)

    @property
    def targets(self):
        return [target for dataset in self.datasets for target in dataset.targets]

    @property
    def classes(self):
        return np.unique(self.targets)


def get_dataset_fold(dataset, *, fold, n_folds, random_state=0):
    if n_folds is None or n_folds <= 1 or fold is None:
        raise ValueError("n_folds must be greater than 1 and fold must be specified")

    from sklearn.model_selection import StratifiedKFold

    data_length = len(dataset)
    targets = dataset.targets

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    skf_splits = list(skf.split(X=np.zeros(data_length), y=targets))
    train_idx, val_idx = skf_splits[fold]

    train_data = MySubset(dataset, train_idx)
    val_data = MySubset(dataset, val_idx)

    return train_data, val_data


def get_dataset_holdout(dataset, *, test_size, random_state=0):
    from sklearn.model_selection import StratifiedShuffleSplit

    data_length = len(dataset)
    targets = dataset.targets

    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    sss_splits = list(sss.split(X=np.zeros(data_length), y=targets))
    train_idx, test_idx = sss_splits[0]

    train_data = MySubset(dataset, train_idx)
    test_data = MySubset(dataset, test_idx)

    return train_data, test_data
