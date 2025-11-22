import time
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from utils.log_util import get_logger
from utils.file_util import load_config, setup_mlflow, load_dataset
from utils.data_util import precompute_splits, inspect_dataframe
from model import MLP
from focal_loss import FocalLoss

logger = get_logger()


def train(model, loss_fn, optimizer, train_loader, val_loader, epochs):
    """Train the model and log per-epoch training and validation losses."""
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

        train_loss = total_loss / len(train_loader)

        # Validation loss
        model.eval()
        val_loss_total = 0.0

        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for X_val, y_val in val_loader:
                logits_val = model(X_val)
                loss_val = loss_fn(logits_val, y_val)
                val_loss_total += loss_val.item()

                # Convert logits → probabilities → predictions
                probs = torch.sigmoid(logits_val).cpu().numpy()
                preds = (probs > 0.5).astype(int)

                all_val_preds.extend(preds)
                all_val_labels.extend(y_val.cpu().numpy())

        val_loss = val_loss_total / len(val_loader)

        # Validation F1
        f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)

        logger.info(
            f"[Epoch {epoch + 1:03d}] "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {f1:.4f}"
        )

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_f1", f1, step=epoch)


def evaluate(model, test_loader):
    """Evaluate model on test data and log all relevant metrics to MLflow."""
    model.eval()
    all_preds, all_trues, all_probs = [], [], []

    with torch.no_grad():
        for X, y in test_loader:
            logits = model(X)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_trues.extend(y.cpu().numpy())

    # Metrics
    accuracy = accuracy_score(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds, zero_division=0)
    precision = precision_score(all_trues, all_preds, zero_division=0)
    recall = recall_score(all_trues, all_preds, zero_division=0)
    cm = confusion_matrix(all_trues, all_preds)
    roc_auc = roc_auc_score(all_trues, all_probs)

    # Logging
    logger.info(f"Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, ROC AUC: {roc_auc:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("roc_auc", roc_auc)

    # Save confusion matrix
    np.savetxt("confusion_matrix.csv", cm, delimiter=",")
    mlflow.log_artifact("confusion_matrix.csv")

    # Save model
    mlflow.pytorch.log_model(model, name="model", input_example=X[:1].numpy())


def run_experiments(config):
    # Load dataset
    logger.info("Loading dataset...")
    df = load_dataset(config["creditcard_csv"])
    inspect_dataframe(df)

    # Precompute all splits and loaders
    logger.info("Dataset loaded. Precomputing splits...")
    splits = precompute_splits(config, df)

    # Compute total number of runs
    model_sizes = {
        "small": (2, 32),
        "medium": (3, 64),
        "large": (4, 128)
    }
    selected_models = {k: v for k, v in model_sizes.items() if k in config["models"]}

    total_runs = 0
    for loss_name in config["losses"]:
        if loss_name == "focal":
            sweep_count = max(len(config.get("focal_gamma", [1])), 1)
        else:
            sweep_count = 1  # CE, WCE, L2 run only once
        total_runs += len(splits) * len(selected_models) * sweep_count
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

        for model_name, (num_layers, hidden_size) in selected_models.items():
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
                    val_name = f"v{val:.1f}"
                    run_name = f"{split_name}_{model_name}_{loss_name}_{val_name}"
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
                        train(model, loss_fn, optimizer, train_loader, val_loader, config["epochs"])
                        evaluate(model, test_loader)

                    elapsed = time.time() - run_start
                    logger.info(f"Finished run {run_idx}/{total_runs} in {elapsed:.1f}s")
    logger.info(f"Finished experiment in {time.time() - start_time_all:.1f}s")


if __name__ == "__main__":
    config = load_config("main_config.yaml")
    #config = load_config("small_config.yaml")
    #config = load_config("medium_config.yaml")
    #config = load_config("large_config.yaml")
    setup_mlflow(config)
    run_experiments(config)
