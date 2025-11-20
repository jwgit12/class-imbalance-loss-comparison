import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Literal
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.data_util import inspect_dataframe, create_new_ratio_df, df_to_loaders
from utils.file_util import load_config, setup_mlflow, load_dataset
from utils.log_util import get_logger
from focal_loss import FocalLoss
from model import MLP

setup_mlflow()
logger = get_logger()
config = load_config()

# TODO: Calculate the total amount of steps/runs to know the progress made
# TODO: Again, make the loading like before to optimize imports and data loading
# TODO: Implement early stopping to avoid crazy overfit
# TODO: Only use two models
# TODO: Only use 10 epochs?

def train(model, loss_fn, optimizer, train_loader, epochs, verbose=True):
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()

            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if verbose:
            logger.info(f"[Epoch {epoch+1:03d}] Loss: {total_loss:.4f}")

        mlflow.log_metric("train_loss", total_loss, step=epoch)


def evaluate(model, test_loader, verbose=True):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X)
            pred = (torch.sigmoid(logits) > 0.5).cpu().numpy()

            preds.extend(pred)
            trues.extend(y.cpu().numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    cm = confusion_matrix(trues, preds)

    if verbose:
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:\n{cm}")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)

    np.savetxt("confusion_matrix.csv", cm, delimiter=",")
    mlflow.log_artifact("confusion_matrix.csv")

    mlflow.pytorch.log_model(model, "model")


def run_experiment(
    df: pd.DataFrame,
    imbalance_ratio: float,
    model_size: Literal["small", "medium", "large"],
    loss_name: Literal["l2", "ce", "wce", "focal"],
    verbose: bool = True
):
    # Create dataset split
    df, meta = create_new_ratio_df(df, imbalance_ratio, verbose=verbose)

    train_loader, val_loader, test_loader, input_dim = df_to_loaders(
        df, batch_size=config["batch_size"]
    )

    model_sizes = {
        "small": (2, 32),
        "medium": (3, 64),
        "large": (4, 128)
    }
    num_layers, hidden_size = model_sizes[model_size]

    # Determine sweep values for the loss
    if loss_name == "l2":
        sweep_values = [0]
    elif loss_name == "ce":
        sweep_values = [1.0]
    elif loss_name == "wce":
        sweep_values = [meta.num_majority / meta.num_minority]
    elif loss_name == "focal":
        sweep_values = config["focal_gamma"]
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    for val in sweep_values:
        run_name = f"ir{meta.imbalance_ratio:.2f}_ds{meta.reduced_by_percent}_{model_size}_{loss_name}_{val}"
        if verbose:
            logger.info(f"Running {run_name}")

        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "dataset_split": meta.reduced_by_percent,
                "model_size": model_size,
                "num_layers": num_layers,
                "hidden_size": hidden_size,
                "loss": loss_name,
                "loss_param": val,
                "lr": config["lr"]
            })

            model = MLP(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers)

            # Loss selection
            if loss_name == "l2":
                loss_fn = nn.MSELoss()
            elif loss_name == "ce":
                loss_fn = nn.BCEWithLogitsLoss()
            elif loss_name == "wce":
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([val], dtype=torch.float32))
            elif loss_name == "focal":
                loss_fn = FocalLoss(alpha=0.5, gamma=val)

            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

            train(model, loss_fn, optimizer, train_loader, config["epochs"], verbose)
            evaluate(model, test_loader, verbose)


def with_loss(df: pd.DataFrame, model_size: Literal["small", "medium", "large"], loss_name: Literal["l2", "ce", "wce", "focal"], verbose=True):
    for imbalance_ratio in config["imbalance_ratios"]:
        run_experiment(df, imbalance_ratio, model_size, loss_name, verbose=verbose)


def with_model(df: pd.DataFrame, model_size: Literal["small", "medium", "large"], verbose=True):
    for loss_name in config["losses"]:
        with_loss(df, model_size, loss_name, verbose=verbose)


if __name__ == "__main__":
    df = load_dataset("creditcard.csv")
    inspect_dataframe(df)

    with_model(df, "small")
    #with_model(df, "medium")
    #with_model(df, "large")


