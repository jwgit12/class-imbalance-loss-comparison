import time
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils.log_util import get_logger
from utils.file_util import load_config, setup_mlflow, load_dataset
from utils.data_util import precompute_splits
from model import MLP
from focal_loss import FocalLoss

logger = get_logger()
config = load_config()
setup_mlflow(config)


def train(model, loss_fn, optimizer, train_loader, epochs, run_idx, total_runs, start_time):
    model.train()
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        mlflow.log_metric("train_loss", total_loss, step=epoch)
        elapsed = time.time() - epoch_start
        eta_epoch = elapsed * (epochs - epoch - 1)
        eta_all = (time.time() - start_time) / (run_idx + 1) * (total_runs - run_idx - 1)
        logger.info(
            f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f} | "
            f"ETA this run: {eta_epoch:.1f}s | ETA all runs: {eta_all/60:.2f}min"
        )


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

    logger.info(f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1", f1)
    np.savetxt("confusion_matrix.csv", cm, delimiter=",")
    mlflow.log_artifact("confusion_matrix.csv")
    mlflow.pytorch.log_model(model, "model")


def run_experiments():
    # Load dataset once
    df = load_dataset(config["creditcard_csv"])
    logger.info("Dataset loaded. Precomputing splits...")

    # Precompute all splits and loaders
    splits = precompute_splits(df)

    # Compute total number of runs
    model_sizes = {
        "small": (2, 32),
        "medium": (3, 64),
        "large": (4, 128)
    }
    total_runs = len(splits) * len(model_sizes) * len(config["losses"]) * max(len(config.get("focal_gamma", [0])), 1)
    run_idx = 0
    start_time_all = time.time()

    # Loop over splits × models × losses
    for split_name, split_data in splits.items():
        train_loader = split_data["train_loader"]
        val_loader = split_data["val_loader"]
        test_loader = split_data["test_loader"]
        input_dim = split_data["input_dim"]
        meta = split_data["meta"]

        # WCE auto weight
        auto_weight = meta.num_majority / meta.num_minority

        for model_name, (num_layers, hidden_size) in model_sizes.items():
            for loss_name in config["losses"]:
                if loss_name == "l2":
                    sweep_values = [0]
                elif loss_name == "ce":
                    sweep_values = [1.0]
                elif loss_name == "wce":
                    sweep_values = [auto_weight]
                elif loss_name == "focal":
                    sweep_values = config["focal_gamma"]
                else:
                    raise ValueError(f"Unknown loss: {loss_name}")

                for val in sweep_values:
                    run_idx += 1
                    run_name = f"{split_name}_{model_name}_{loss_name}_{val}"
                    logger.info(f"\n--- Run {run_idx}/{total_runs}: {run_name} ---")
                    run_start = time.time()

                    with mlflow.start_run(run_name=run_name):
                        mlflow.log_params({
                            "dataset_split": split_name,
                            "model_size": model_name,
                            "num_layers": num_layers,
                            "hidden_size": hidden_size,
                            "loss": loss_name,
                            "loss_param": val,
                            "lr": config["lr"]
                        })

                        # Build model
                        model = MLP(input_dim=input_dim, hidden_size=hidden_size, num_layers=num_layers)

                        # Select loss
                        if loss_name == "l2":
                            loss_fn = nn.MSELoss()
                        elif loss_name == "ce":
                            loss_fn = nn.BCEWithLogitsLoss()
                        elif loss_name == "wce":
                            loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([val], dtype=torch.float32))
                        elif loss_name == "focal":
                            loss_fn = FocalLoss(alpha=0.5, gamma=val)

                        # Optimizer
                        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

                        # Train and evaluate
                        train(model, loss_fn, optimizer, train_loader, config["epochs"], run_idx, total_runs, start_time_all)
                        evaluate(model, test_loader)

                    elapsed = time.time() - run_start
                    logger.info(f"Finished run {run_idx}/{total_runs} in {elapsed:.1f}s")


if __name__ == "__main__":
    run_experiments()
