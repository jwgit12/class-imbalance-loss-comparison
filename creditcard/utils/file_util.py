from pathlib import Path

import mlflow
import pandas as pd
import yaml

from .log_util import get_logger

logger = get_logger()

# Resolve project root (directory ABOVE utils/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(name:str) -> dict:
    config_path = PROJECT_ROOT / "configs" / name

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    logger.info("Loading config...")
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def load_dataset(filename: str) -> pd.DataFrame:
    dataset_path = PROJECT_ROOT / "data" / filename
    logger.info("Loading dataset...")
    dataset = pd.read_csv(fr"{dataset_path}")
    return dataset


def setup_mlflow(config):
    mlflow.set_tracking_uri("http://localhost:" + config["mlflow_port"])
    mlflow.set_experiment(config["mlflow_experiment_name"])
