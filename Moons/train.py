"""
train.py
========

This script serves as the main entry point for training and evaluating a simple
MLP classifier on the Moons dataset using three different loss functions:

 - Binary Cross-Entropy (CE)
 - Weighted Cross-Entropy (WCE)
 - Focal Loss

The script loads settings from `experiment_config.yaml`, builds the dataset,
initializes selected models & loss functions, trains them, and evaluates results.
"""
import logging

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from dataset import MoonsDataset
from focal_loss import FocalLoss
from model import MLP
from utils import load_config

# -----------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

# -----------------------------------------------------------
# MLfLow configuration
# -----------------------------------------------------------
def setup_mlflow(config):
    mlflow.set_tracking_uri("http://localhost:"+config["mlflow_port"])
    mlflow.set_experiment(config["mlflow_experiment_name"])

# -----------------------------------------------------------
# Dataset creation and dataloader preparation
# -----------------------------------------------------------
def prepare_dataloaders(config):
    """
    Creates dataset and splits it into train/validation/test sets.

    Args:
        config (dict): Experiment configuration dictionary.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    dataset = MoonsDataset(
        n_samples=config["num_samples"],
        noise=config["noise"],
        imbalance_ratio=config["imbalance_ratio"][0],         # ignored when balanced=True
        random_state=config["random_seed"]
    )

    # Generate index split
    indices = np.arange(len(dataset))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=config["test_size"],
        random_state=config["random_seed"]
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=config["val_size"],
        random_state=config["random_seed"]
    )

    # Dataloaders
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=config["batch_size"])
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=config["batch_size"])

    return train_loader, val_loader, test_loader

# -----------------------------------------------------------
# Training loop
# -----------------------------------------------------------
def train(model, loss_fn, optimizer, train_loader, epochs):
    """
    Trains the model for a given number of epochs.

    Args:
        model (nn.Module)
        loss_fn (callable)
        optimizer (torch.optim.Optimizer)
        train_loader (DataLoader)
        epochs (int)
    """
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0.0

        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        logger.info(f"[Epoch {epoch+1:03d}] Loss: {epoch_loss:.4f}")
        mlflow.log_metric("train_loss", epoch_loss, step=epoch)

# -----------------------------------------------------------
# Evaluation function
# -----------------------------------------------------------
def evaluate(model, test_loader):
    """
    Evaluates the trained model on the test split.

    Args:
        model (nn.Module)
        test_loader (DataLoader)
    """
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

    logger.info(f"Accuracy : {acc:.4f}")
    logger.info(f"F1 Score : {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    np.savetxt("confusion_matrix.csv", cm, delimiter=",")
    mlflow.log_artifact("confusion_matrix.csv")

    # Save the model
    mlflow.pytorch.log_model(model, name="model")

# -----------------------------------------------------------
# Experiment loop: run CE, WCE, and Focal Loss
# -----------------------------------------------------------
def run_experiments(config):
    """
    Runs an experiment for each loss function specified in the config.

    Args:
        config (dict): Experiment configuration.
    """
    train_loader, val_loader, test_loader = prepare_dataloaders(config)

    for loss_name in config["losses"]:
        with mlflow.start_run(run_name=f"{loss_name.upper()}_run"):
            logger.info(f" Training with loss: {loss_name}")

            mlflow.log_param("loss_fn", loss_name)
            mlflow.log_param("hidden_size", config["hidden_size"])
            mlflow.log_param("num_layers", config["num_layers"])
            mlflow.log_param("lr", config["lr"])
            mlflow.log_param("batch_size", config["batch_size"])
            mlflow.log_param("noise", config["noise"])
            mlflow.log_param("num_samples", config["num_samples"])
            mlflow.log_param("imbalance_ratio", config["imbalance_ratio"][0])


            # Build model
            model = MLP(
                input_dim=2,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"]
            )

            # Select loss
            if loss_name == "ce":
                loss_fn = nn.BCEWithLogitsLoss()

            elif loss_name == "wce":
                # Use first provided weight for demonstration
                pos_weight = torch.tensor([config["ce_weights"][0]])
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            elif loss_name == "focal":
                loss_fn = FocalLoss(
                    alpha=config["focal_alpha"][0],
                    gamma=config["focal_gamma"][0]
                )

            # Optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

            # Train + Eval
            train(model, loss_fn, optimizer, train_loader, config["epochs"])
            evaluate(model, test_loader)

if __name__ == "__main__":
    """
    Run with `python train.py`.
    """
    config = load_config()
    setup_mlflow(config)
    run_experiments(config)
