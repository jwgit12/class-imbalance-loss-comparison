import logging
import mlflow
import mlflow.pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from focal_loss import FocalLoss
from model import MLP
from utils import (
    load_config,
    setup_mlflow,
    create_imbalanced_splits,
    df_to_loaders, inspect_creditcard_dataframe,
)

# -----------------------------------------------------------
# Logging
# -----------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)


# -----------------------------------------------------------
# Training loop
# -----------------------------------------------------------
def train(model, loss_fn, optimizer, train_loader, epochs):
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

        logger.info(f"[Epoch {epoch+1:03d}] Loss: {total_loss:.4f}")
        mlflow.log_metric("train_loss", total_loss, step=epoch)


# -----------------------------------------------------------
# Evaluation loop
# -----------------------------------------------------------
def evaluate(model, test_loader):
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

    logger.info(f"Accuracy: {acc:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)

    np.savetxt("confusion_matrix.csv", cm, delimiter=",")
    mlflow.log_artifact("confusion_matrix.csv")

    mlflow.pytorch.log_model(model, "model")


# -----------------------------------------------------------
# Main experiment loop
# -----------------------------------------------------------
def run_experiments(config):
    # -------------------------------------------------------
    # Load full dataset from Kaggle
    # -------------------------------------------------------
    df = pd.read_csv(r".\data\creditcard.csv")
    inspect_creditcard_dataframe(df)

    # -------------------------------------------------------
    # Create imbalance variants
    # -------------------------------------------------------
    drop_ratios = config["imbalance_profiles"]
    datasets = create_imbalanced_splits(df, drop_ratios)

    # -------------------------------------------------------
    # Model sizes
    # -------------------------------------------------------
    model_sizes = {
        "small":  (1, 16),
        "medium": (2, 64),
        "large":  (3, 128)
    }

    # -------------------------------------------------------
    # Run experiment for each dataset variant
    # -------------------------------------------------------
    for split_name, df_split in datasets.items():
        logger.info(f"\n\n===== Using dataset split: {split_name} =====")
        logger.info(df_split["Class"].value_counts())

        # Build loaders for this dataset version
        train_loader, val_loader, test_loader, input_dim = df_to_loaders(
            df_split,
            batch_size=config["batch_size"]
        )

        # ----------------------------------------
        # Auto weight for WCE: majority/minority
        # ----------------------------------------
        majority = (df_split["Class"] == 0).sum()
        minority = (df_split["Class"] == 1).sum()
        auto_weight = majority / minority

        # -------------------------------------------------------
        # Loop: model size × loss function × sweep values
        # -------------------------------------------------------
        for model_name, (layers, hidden) in model_sizes.items():
            for loss_name in config["losses"]:

                # ----------------------------------------
                # Loss sweeps (per-loss hyperparameters)
                # ----------------------------------------
                if loss_name == "l2":
                    sweep_values = [0]  # no sweep
                elif loss_name == "ce":
                    sweep_values = [1.0]
                elif loss_name == "wce":
                    sweep_values = [auto_weight]  # *important*
                elif loss_name == "focal":
                    sweep_values = config["focal_gamma"]
                else:
                    raise ValueError(f"Unknown loss: {loss_name}")

                # ----------------------------------------
                # For each hyperparameter value
                # ----------------------------------------
                for val in sweep_values:
                    run_name = f"{split_name}_{model_name}_{loss_name}_{val}"

                    with mlflow.start_run(run_name=run_name):
                        mlflow.log_param("dataset_split", split_name)
                        mlflow.log_param("model_size", model_name)
                        mlflow.log_param("loss", loss_name)
                        mlflow.log_param("loss_param", val)
                        mlflow.log_param("hidden_size", hidden)
                        mlflow.log_param("num_layers", layers)
                        mlflow.log_param("lr", config["lr"])

                        # ---------------------
                        # Build model
                        # ---------------------
                        model = MLP(
                            input_dim=input_dim,
                            hidden_size=hidden,
                            num_layers=layers
                        )

                        # ---------------------
                        # Configure loss
                        # ---------------------
                        if loss_name == "l2":
                            loss_fn = nn.MSELoss()

                        elif loss_name == "ce":
                            loss_fn = nn.BCEWithLogitsLoss()

                        elif loss_name == "wce":
                            pos_weight = torch.tensor([val], dtype=torch.float32)
                            loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                        elif loss_name == "focal":
                            loss_fn = FocalLoss(alpha=0.5, gamma=val)

                        # ---------------------
                        # Optimizer
                        # ---------------------
                        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

                        # ---------------------
                        # Train + Eval
                        # ---------------------
                        train(model, loss_fn, optimizer, train_loader, config["epochs"])
                        evaluate(model, test_loader)


# -----------------------------------------------------------
# Run main
# -----------------------------------------------------------
if __name__ == "__main__":
    config = load_config()
    setup_mlflow(config)
    run_experiments(config)
