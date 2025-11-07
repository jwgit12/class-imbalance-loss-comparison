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
from pathlib import Path

config_path = Path(__file__).parent / "experiment_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)


version: str = config["version"]
dataset: str = config["dataset"]
random_seed: int = config["random_seed"]
num_samples: int = config["num_samples"]
noises: list[float] = config["noises"]
imbalance_ratios: list[float] = config["imbalance_ratios"]
model_name: str = config["model"]
num_layers: int = config["num_layers"]
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

for noise, imbalance_ratio, loss_fn in itertools.product(noises, imbalance_ratios, loss_functions):
    print(f"Running Experiment with Noise: {noise}, Imbalance Ratio {imbalance_ratio}, Loss Function: {loss_fn}")
    # Initialize dataset
    if loss_fn == "focal":
        param_combinations = itertools.product(focal_alpha, focal_gamma)
    elif loss_fn == "wce":
        param_combinations = [(weight,) for weight in ce_weights]
    else:  # "ce"
        param_combinations = [(None,)]
    for params in param_combinations:
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


        # Initialize model, optimizer, and loss function
        model: MLP = MLP(input_size=2, hidden_size=hidden_size, num_layers=num_layers)
        optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=lr)
        if loss_fn == "ce":
            criterion = nn.BCEWithLogitsLoss()
        elif loss_fn == "wce":
            weight = params[0]
            criterion: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
        else:
            # Add Focal Loss here
            alpha, gamma = params
            break

        # Training loop
        losses: list[float] = []

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
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

        # Validation
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

        # Compute ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc: float = auc(fpr, tpr)

        print("Validation Metrics:")
        print("=" * 50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        break
    break