import copy
from pathlib import Path
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import yaml
import mlflow

from model import MLP
from focal_loss import FocalLoss

# Load configuration
config_path = Path(__file__).parent / "experiment_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

config_version: str = config["version"]
dataset_name: str = config["dataset"]
random_seed: int = config["random_seed"]
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

mlflow.set_experiment("New Creditcard test")

# --- Load Credit Card Dataset ---
csv_path = Path(__file__).parent / "creditcard.csv"
df = pd.read_csv(csv_path)

# Assuming the label column is named 'Class'
y_labels = df['Class'].values
X_data = df.drop(columns=['Class']).values

# Convert to torch tensors
X_tensor = torch.tensor(X_data, dtype=torch.float32)
y_tensor = torch.tensor(y_labels, dtype=torch.long)  # BCEWithLogitsLoss expects float later

# Wrap in a TensorDataset
full_dataset = TensorDataset(X_tensor, y_tensor)

indices = np.arange(len(full_dataset))

for loss_fn, num_layer in itertools.product(loss_functions, num_layers):
    # Param combinations for losses
    if loss_fn == "focal":
        param_combinations = itertools.product(focal_alpha, focal_gamma)
    elif loss_fn == "wce":
        param_combinations = [(weight,) for weight in ce_weights]
    else:  # "ce"
        param_combinations = [(None,)]

    # --- Train-Test-Val Split ---
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=y_labels,
        random_state=random_seed
    )

    train_val_labels = y_labels[train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size / (1 - test_size),
        stratify=train_val_labels,
        random_state=random_seed
    )

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    test_dataset = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for params in param_combinations:
        print(f"Starting experiment: num_layer={num_layer}, loss_fn={loss_fn}, params={params}")
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("config_version", config_version)
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("num_layer", num_layer)
            mlflow.log_param("hidden_size", hidden_size)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", lr)
            mlflow.log_param("epochs", epochs)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("loss_function", loss_fn)

            # Initialize model and optimizer
            input_size = X_data.shape[1]
            model = MLP(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            # Initialize loss
            if loss_fn == "ce":
                criterion = nn.BCEWithLogitsLoss()
            elif loss_fn == "wce":
                weight = params[0]
                criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
                mlflow.log_param("ce_weight", weight)
            elif loss_fn == "focal":
                alpha, gamma = params
                mlflow.log_param("alpha", alpha)
                mlflow.log_param("gamma", gamma)
                criterion = FocalLoss(alpha=alpha, gamma=gamma)
            else:
                raise ValueError("Loss type not defined")

            # --- Training Loop ---
            best_val_f1 = 0.0
            best_model_state = None
            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y.float().unsqueeze(1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                epoch_loss = total_loss / len(train_loader)

                # --- Validation ---
                model.eval()
                val_loss = 0.0
                all_preds, all_labels_list = [], []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y.float().unsqueeze(1))
                        val_loss += loss.item()
                        probs = torch.sigmoid(outputs)
                        binary_preds = (probs > 0.5).int().squeeze()
                        all_labels_list.extend(batch_y.cpu().numpy())
                        all_preds.extend(binary_preds.cpu().numpy())
                val_loss /= len(val_loader)
                all_labels_list = np.array(all_labels_list)
                all_preds = np.array(all_preds)
                val_f1 = f1_score(all_labels_list, all_preds)
                print(f"Epoch {epoch+1}/{epochs}, Val F1: {val_f1:.4f}")

                # Log metrics
                mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = copy.deepcopy(model.state_dict())
                    print(f"New best F1: {best_val_f1:.4f}")

            # --- Test Evaluation ---
            if best_model_state:
                model.load_state_dict(best_model_state)
            model.eval()
            all_preds, all_probs, all_labels_list = [], [], []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x)
                    probs = torch.sigmoid(outputs.squeeze())
                    preds = (probs > 0.5).int()
                    all_preds.extend(preds.numpy())
                    all_probs.extend(probs.numpy())
                    all_labels_list.extend(batch_y.numpy().astype(int))

            accuracy = accuracy_score(all_labels_list, all_preds)
            precision = precision_score(all_labels_list, all_preds)
            recall = recall_score(all_labels_list, all_preds)
            f1 = f1_score(all_labels_list, all_preds)
            cm = confusion_matrix(all_labels_list, all_preds)

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)

            # Log the model
            mlflow.pytorch.log_model(model, name="MLP", input_example=X_tensor[:1].numpy())
            print(f"Test metrics -> Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
