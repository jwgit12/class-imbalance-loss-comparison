import io
import zipfile
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

# -----------------------------------------------------------
# Load config from YAML
# -----------------------------------------------------------
def load_config():
    """
    Loads the experiment configuration from `experiment_config.yaml`.

    Returns:
        dict: Parsed configuration values.
    """
    config_path = Path(__file__).parent / "experiment_config.yaml"
    with config_path.open("r") as f:
        return yaml.safe_load(f)

# -----------------------------------------------------------
# Split dataset to ratio
# -----------------------------------------------------------
def create_imbalanced_splits(df, drop_ratios):
    splits = {}
    df_maj = df[df["Class"] == 0]
    df_min = df[df["Class"] == 1]

    for ratio in drop_ratios:
        keep_n = int(len(df_maj) * ratio)
        df_maj_new = df_maj.sample(keep_n, random_state=42)
        df_new = pd.concat([df_maj_new, df_min], axis=0).sample(frac=1, random_state=42)
        name = f"r{int(ratio * 100)}"
        splits[name] = df_new

    return splits


def df_to_loaders(df, batch_size):
    X = df.drop("Class", axis=1).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    def make_loader(X, y):
        tensor_X = torch.tensor(X)
        tensor_y = torch.tensor(y)
        dataset = TensorDataset(tensor_X, tensor_y)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_loader = make_loader(X_train, y_train)
    val_loader = make_loader(X_val, y_val)
    test_loader = make_loader(X_test, y_test)

    return train_loader, val_loader, test_loader, X.shape[1]

def inspect_creditcard_dataframe(df, name="Dataset"):
    print(f"\n--- Inspecting {name} ---\n")

    # Shape
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Head
    print("Head:")
    print(df.head(), "\n")

    # Missing values
    missing = df.isnull().sum().sum()
    print(f"Missing values: {missing}\n")

    # Class distribution
    print("Class distribution:")
    counts = df["Class"].value_counts()
    print(counts, "\n")

    # Imbalance ratio
    if 0 in counts.index and 1 in counts.index:
        maj = counts[0]
        mino = counts[1]
        ratio = maj / mino
        print(f"Imbalance ratio (majority/minority): {maj} / {mino} = {ratio:.1f} : 1\n")

    # Basic statistics for numerical features
    #print("Feature statistics (numeric columns):")
    #print(df.describe().transpose(), "\n")

# -----------------------------------------------------------
# Setup MLflow
# -----------------------------------------------------------
def setup_mlflow(config):
    mlflow.set_tracking_uri("http://localhost:" + config["mlflow_port"])
    mlflow.set_experiment(config["mlflow_experiment_name"])

