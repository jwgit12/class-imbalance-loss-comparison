import os
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from creditcard.utils.file_util import load_config, setup_mlflow


def get_loss_label(row):
    loss = row["params.loss"]

    if loss == "focal":
        gamma = str(row["params.loss_param"])
        key = f"focal_{gamma}"
        return key, LOSS_COLORS[key]
    else:
        return loss, LOSS_COLORS[loss]


def get_style_for_split(row):
    split = row["params.dataset_split"]
    return RATIO_STYLES[split]

def get_marker_for_loss(row):
    loss_type = row["params.loss"]
    return LOSS_MARKERS.get(loss_type, "o")

def get_name_for_ratio(split):
        return RATIO_NAMES[split]

config = load_config("main_config.yaml")
setup_mlflow(config)

def load_runs():
    """Load runs for the configured experiment."""
    client: MlflowClient = MlflowClient()
    experiment = client.get_experiment_by_name(config["mlflow_experiment_name"])

    df: pd.DataFrame = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    return df, client


PLOT_DIR = "plots"
LOSS_COLORS = {
    "ce": "#1f77b4",
    "wce": "#ff7f0e",
    "focal_0.0": "#d62728",
    "focal_1.0": "#d62728",
    "focal_2.0": "#b22222",
    "focal_5.0": "#ff9999",
}
RATIO_STYLES = {
    "r100": "-",
    "r60": "--",
    "r30": ":",
    "r0": "-.",
}
LOSS_MARKERS = {
    "ce": "o",
    "wce": "s",
    "focal": "^",
}
RATIO_NAMES = {
    "r100": "No reduction - fully imbalanced",
    "r60": "60% of majority dataset",
    "r30": "30% of majority dataset",
    "r0": "Equally Balanced",
}
