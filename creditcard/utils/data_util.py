from typing import Tuple, Dict

import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
import torch
import numpy as np

from .file_util import load_config
from .log_util import get_logger

logger = get_logger()
config = load_config()


@dataclass
class MetaData:
    num_majority: int # Number of majority samples
    num_minority: int # number of minority samples
    reduced_by_percent: int # % of reduction of majority samples
    imbalance_ratio: float # Ratio of the imbalance after reducing

def precompute_splits(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Precompute train/val/test loaders for all imbalance ratios.
    Returns dictionary keyed by ratio name (r100, r60, etc.)
    Each entry contains: train_loader, val_loader, test_loader, meta
    """
    splits = {}
    df_maj = df[df["Class"] == 0]
    df_min = df[df["Class"] == 1]

    for ratio in config["imbalance_ratios"]:
        keep_n = int(len(df_maj) * ratio) if ratio > 0 else len(df_min)
        new_maj_df = df_maj.sample(keep_n, random_state=config["random_seed"])
        new_df = pd.concat([df_min, new_maj_df], axis=0).sample(frac=1, random_state=config["random_seed"])
        reduced_percent = int(ratio * 100)

        meta = MetaData(
            num_majority=len(new_maj_df),
            num_minority=len(df_min),
            reduced_by_percent=reduced_percent,
            imbalance_ratio=len(new_maj_df) / len(df_min)
        )

        train_loader, val_loader, test_loader, input_dim = df_to_loaders(new_df, config["batch_size"])

        split_name = f"r{reduced_percent}"
        splits[split_name] = {
            "train_loader": train_loader,
            "val_loader": val_loader,
            "test_loader": test_loader,
            "input_dim": input_dim,
            "meta": meta
        }

        logger.info(f"Precomputed split {split_name}: {meta.num_majority} majority, {meta.num_minority} minority")

    return splits

def create_new_ratio_df(original_df: pd.DataFrame, drop_ratio: float, verbose = False) -> Tuple[pd.DataFrame, MetaData]:
    """
    Create a new DataFrame with a downsampled majority class (class 0).

    :param original_df: Original dataset containing a binary column named "Class".
    :param drop_ratio: Fraction of majority samples (class 0) to keep. Must be between 0 and 1.
    If the drop_ratio is 0, the split will be 1:1 wit as much majority samples as minority samples.
    :param verbose: If `True` log information to the console. `False` is default.

    :return: Tuple[pd.DataFrame, int, int, int]:
            pd.DataFrame: The downsampled dataset containing all minority samples and the reduced majority class.
            int: Number of majority class samples in the new DataFrame.
            int: Number of minority class samples (unchanged).
            int: Percentage of majority class samples removed (0–100).
    """

    # Split into its majority and minority classes
    df_maj = original_df[original_df["Class"] == 0]
    df_min = original_df[original_df["Class"] == 1]

    # Determine amount to keep from the majority samples
    # If drop_ratio is 0,
    keep_n = int(len(df_maj) * drop_ratio) if drop_ratio > 0 else len(df_min)


    # Create a DataFrame with the new amount of samples to keep
    new_maj_df = df_maj.sample(keep_n, random_state=config["random_seed"])

    # Concat the new majority and minority samples and shuffle them
    new_df = pd.concat([df_min, new_maj_df], axis=0).sample(frac=1, random_state=config["random_seed"])

    meta = MetaData(
        num_majority=len(new_maj_df),
        num_minority=len(df_min),
        reduced_by_percent=int(drop_ratio * 100),
        imbalance_ratio=(len(new_maj_df) / len(df_min)),
    )

    if verbose:
        logger.info(f"\n\n--- Inspecting Ratio Adjustment ---")
        logger.info(f"Remove {100 - (drop_ratio * 100)}% of the majority samples: {len(new_maj_df)}, keep: {keep_n}")
        logger.info(f"Original: maj: {len(df_maj)},     min: {len(df_min)} -> {(len(df_min)/len(df_maj) * 100):.2f}%")
        logger.info(f"New:      maj: {len(new_maj_df)},     min: {len(df_min)} -> {(len(df_min)/len(new_maj_df) * 100):.2f}%")
        logger.info(f"Old Imbalance Ratio: {(len(df_maj) / len(df_min)):.2f}")
        logger.info(f"New Imbalance Ratio: {(len(new_maj_df) / len(df_min)):.2f}")
        logger.info(f"Len original dataset: {len(original_df)}")
        logger.info(f"Len new dataset: {len(new_df)}")

    return new_df, meta


def df_to_loaders(df, batch_size):
    logger.info("Transforming DataFrames into loaders...")

    X = df.drop("Class", axis=1).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=config["random_seed"])
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=config["random_seed"])

    def make_loader(X, y):
        tensor_X = torch.tensor(X)
        tensor_y = torch.tensor(y)
        dataset = TensorDataset(tensor_X, tensor_y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    return train_loader, val_loader, test_loader, X.shape[1]


def inspect_dataframe(df: pd.DataFrame):
    """
    Takes a `pandas.DataFrame` and inspects some key features such as:
        - shape
        - show head
        - missing values
    :param df: Dataset to inspect
    """

    logger.info(f"\n\n--- Inspecting Dataset ---")
    logger.info(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"{df.head()}")

    missing = df.isnull().sum().sum()
    logger.info(f"Missing values: {missing}")