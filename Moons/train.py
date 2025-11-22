import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
from model import MLP
from dataset import ImbalancedMoonsDataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import itertools
import yaml
import copy
from pathlib import Path
import mlflow
from focal_loss import FocalLoss

# TODO: Early Stopping, Use of best model on Val Loss.
config_path = Path(__file__).parent / "experiment_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


config_version: str = config["version"]
dataset_name: str = config["dataset"]
random_seed: int = config["random_seed"]
num_samples: int = config["num_samples"]
noises: list[float] = config["noises"]
imbalance_ratios: list[float] = config["imbalance_ratios"]
model_name: str = config["model"]
num_layers: list[int] = config["num_layers"]
hidden_size: int = config["hidden_size"]
epochs: int = config["epochs"]
batch_size: int = config["batch_size"]
test_size: float = config["test_size"]
val_size: float = config["val_size"]
lr: float = config["lr"]
loss_functions: list[str] = config["losses"]
focal_alpha: list[float] = config["focal_alpha"]
focal_gamma: list[float] = config["focal_gamma"]
ce_weights: list[float] = config["ce_weights"]

mlflow.set_experiment("Imbalanced Moons Classification")
#mlflow.log_artifact(str(config_path), artifact_path="config")
for noise, imbalance_ratio, loss_fn, num_layer in itertools.product(noises, imbalance_ratios, loss_functions, num_layers):
    # Initialize dataset
    if loss_fn == "focal":
        param_combinations = itertools.product(focal_alpha, focal_gamma)
    elif loss_fn == "wce":
        param_combinations = [(weight,) for weight in ce_weights]
    else:  # "ce"
        param_combinations = [(None,)]

    # Create dataset
    dataset: ImbalancedMoonsDataset = ImbalancedMoonsDataset(
        n_samples=num_samples,
        noise=noise,
        imbalance_ratio=imbalance_ratio,
        random_state=random_seed
    )

    # Train-test split with stratification
    indices: np.ndarray = np.arange(len(dataset))
    y_labels: np.ndarray = dataset.y.numpy()

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y_labels,
        random_state=random_seed
    )

    # Second split: train vs val from train_val set
    train_val_labels = y_labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        stratify=train_val_labels,
        random_state=random_seed
    )

    # Create train, val, and test loaders
    train_dataset: Subset = Subset(dataset, train_idx)
    val_dataset: Subset = Subset(dataset, val_idx)
    test_dataset: Subset = Subset(dataset, test_idx)

    train_loader: DataLoader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader: DataLoader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader: DataLoader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for params in param_combinations:
        print(f"Starting experiment with noise={noise}, imbalance_ratio={imbalance_ratio}, num_layer={num_layer}, loss_fn={loss_fn}, params={params}")
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("config_version", config_version)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("num_samples", num_samples)
            mlflow.log_param("noise", noise)
            mlflow.log_param("imbalance_ratio", imbalance_ratio)
            mlflow.log_param("loss_function", loss_fn)
            mlflow.log_param("num_layer", num_layer)
            mlflow.log_param("hidden_size", hidden_size)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("val_size", val_size)



            # Initialize model, optimizer, and loss function
            model: MLP = MLP(input_size=2, hidden_size=hidden_size, num_layers=num_layer)
            optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
            if loss_fn == "ce":
                criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
            elif loss_fn == "wce":
                weight: float = params[0]
                criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
                mlflow.log_param("ce_weight", weight)
            elif loss_fn == "focal":
                # Add Focal Loss here
                alpha, gamma = params
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("gamma", gamma)
                criterion: FocalLoss = FocalLoss(gamma = gamma, alpha= alpha)  # type: ignore[no-redef]
            else:
                raise ValueError("Loss Type not defined")

            # Training loop
            losses: list[float] = []
            best_val_loss: float = float("inf")
            best_val_f1: float = 0.0
            best_model_state = None
            for epoch in range(epochs):
                model.train()
                total_loss: float = 0.0

                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs: torch.Tensor = model(batch_x)
                    loss: torch.Tensor = criterion(outputs, batch_y.float().unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                epoch_loss: float = total_loss / len(train_loader)
                losses.append(epoch_loss)
                # Validate Epoch
                model.eval()
                val_loss = 0.0
                all_preds = []
                all_labels = []

                # --- Validation Loop ---
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        # Note: No explicit .to(device) or .cuda() is needed here

                        outputs = model(batch_x)

                        # 1. Calculate validation loss
                        loss = criterion(outputs, batch_y.float().unsqueeze(1))
                        val_loss += loss.item()

                        # 2. Convert outputs to binary predictions
                        # Assuming a binary classification where the final layer is Sigmoid
                        probs = torch.sigmoid(outputs)

                        # Apply a threshold (e.g., 0.5) to get binary predictions (0 or 1)
                        # Squeeze(1) is used to remove the dimension of size 1 if present
                        binary_preds = (probs > 0.5).int().squeeze()

                        # 3. Store true labels and predictions
                        # Move tensors to CPU before converting to NumPy
                        all_labels.extend(batch_y.cpu().numpy())
                        all_preds.extend(binary_preds.cpu().numpy())

                val_loss /= len(val_loader)

                # Convert lists to NumPy arrays
                all_labels = np.array(all_labels)
                all_preds = np.array(all_preds)

                # 4. Calculate the F1 score for the entire validation set
                val_f1 = f1_score(all_labels, all_preds)
                print(f"Validation F1 Score: {val_f1:.4f}")

                # --- Model Selection and Logging ---
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)

                # 5. Model Selection based on F1 Score (Higher is better)
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    # Save the model state as it achieved the highest F1 score so far
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f"New best F1 score: {best_val_f1:.4f}. Model state updated.")

            # Validation
            if best_model_state:
                model.load_state_dict(best_model_state)
            model.eval()
            all_preds: list[int] = []
            all_probs: list[float] = []
            all_labels: list[int] = []

            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs: torch.Tensor = model(batch_x)
                    probs: torch.Tensor = torch.sigmoid(outputs.squeeze())
                    preds: torch.Tensor = (probs > 0.5).int()
                    all_preds.extend(preds.numpy())
                    all_probs.extend(probs.numpy())
                    all_labels.extend(batch_y.numpy().astype(int))

            # Calculate metrics
            accuracy: float = accuracy_score(all_labels, all_preds)
            precision: float = precision_score(all_labels, all_preds)
            recall: float = recall_score(all_labels, all_preds)
            f1: float = f1_score(all_labels, all_preds)
            cm: np.ndarray = confusion_matrix(all_labels, all_preds)

            # Log metrics after training
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log the model
            print(batch_x[:1].numpy().shape)
            mlflow.pytorch.log_model(model, name="MLP", input_example=batch_x[:1].numpy())