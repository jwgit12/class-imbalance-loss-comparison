import copy
import time
from pathlib import Path
import itertools
import logging

import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sympy.plotting.experimental_lambdify import experimental_lambdify
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import yaml
import mlflow

from model import MLP
from focal_loss import FocalLoss

# --------------------------
# Logging setup
# --------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --------------------------
# Load configuration
# --------------------------
config_path = Path(__file__).parent / "experiment_config.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# --------------------------
# Load dataset
# --------------------------
csv_path = Path(__file__).parent / "creditcard.csv"
df = pd.read_csv(csv_path)

X_full = df.drop(columns=["Class"]).values.astype(np.float32)
y_full = df["Class"].values.astype(np.float32)  # BCE expects float

mlflow.set_tracking_uri("http://localhost:" + "32770")
mlflow.set_experiment("CF_r1.0_0.5_0.1_e50_b1024_v3")

# --------------------------
# Function to create stratified loaders
# --------------------------
def create_loaders(X, y, batch_size, test_size=0.2, val_size=0.1, random_seed=42):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_seed
    )

    val_frac = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, stratify=y_trainval, random_state=random_seed
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def make_loader(X_arr, y_arr, shuffle=True):
        dataset = TensorDataset(torch.tensor(X_arr), torch.tensor(y_arr))
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return (
        make_loader(X_train, y_train),
        make_loader(X_val, y_val, shuffle=False),
        make_loader(X_test, y_test, shuffle=False),
    )

# Count total number of runs
total_runs = len(config["imbalance_ratios"]) * len(config["num_layers"]) * \
             sum(len(config["focal_alpha"]) * len(config["focal_gamma"]) if l == "focal"
                 else len(config["ce_weights"]) if l == "wce"
                 else 1
                 for l in config["losses"])

current_run = 0

# --------------------------
# Training loop over imbalance ratios and model configs
# --------------------------
for ratio in config["imbalance_ratios"]:
    df_min = df[df["Class"] == 1]
    df_maj = df[df["Class"] == 0]
    keep_n = int(len(df_maj) * ratio)
    df_maj_sampled = df_maj.sample(keep_n, random_state=config["random_seed"])
    df_sub = pd.concat([df_min, df_maj_sampled], axis=0).sample(frac=1, random_state=config["random_seed"])

    X = df_sub.drop(columns=["Class"]).values.astype(np.float32)
    y = df_sub["Class"].values.astype(np.float32)

    train_loader, val_loader, test_loader = create_loaders(
        X, y, batch_size=config["batch_size"], test_size=config["test_size"], val_size=config["val_size"], random_seed=config["random_seed"]
    )

    input_dim = X.shape[1]
    num_minority = len(df_min)
    num_majority = len(df_maj_sampled)
    imbalance_ratio = num_majority / num_minority

    experiment_start_time = time.time()
    for num_layer, loss_fn in itertools.product(config["num_layers"], config["losses"]):
        if loss_fn == "focal":
            param_combinations = itertools.product(config["focal_alpha"], config["focal_gamma"])
        elif loss_fn == "wce":
            param_combinations = [(w,) for w in config["ce_weights"]]
        else:
            param_combinations = [(None,)]

        for params in param_combinations:
            current_run += 1
            print("="*70)
            print(f"RUN {current_run}/{total_runs} STARTED")
            print(f"{ratio*100}% samples from majority class are taken. \n"
                  f"Majority: {num_majority}, Minority: {num_minority}, Ratio: {(num_majority/num_minority):.2f}")
            if loss_fn == "focal":
                alpha, gamma = params
                params_str = f"a{alpha}, g{gamma}"
            elif loss_fn == "wce":
                weight = params[0]
                params_str = f"w{weight}"
            else:
                params_str = params[0] # = None
            print(f"Loss: {loss_fn}, Loss params: {params_str}, Num Layers: {num_layer}, Hidden Size: {config['hidden_size']}")
            print("="*70)

            # Build a descriptive run name
            loss_param_str = ""
            if loss_fn == "wce":
                loss_param_str = f"w{params[0]}"
            elif loss_fn == "focal":
                loss_param_str = f"a{params[0]}_g{params[1]}"
            else:
                loss_param_str = "0"
            r2=f"{(num_majority / num_minority):.2f}"
            run_name = f"r{ratio}={r2}_{loss_fn}_{loss_param_str}"

            run_start_time = time.time()
            with mlflow.start_run(run_name=run_name):
                # Log params
                mlflow.log_param("imbalance_ratio", imbalance_ratio)
                mlflow.log_param("subsample_ratio", ratio)
                mlflow.log_param("num_majority", num_majority)
                mlflow.log_param("num_minority", num_minority)
                mlflow.log_param("loss_function", loss_fn)
                mlflow.log_param("loss_params", str(params))
                mlflow.log_param("num_layers", num_layer)
                mlflow.log_param("hidden_size", config["hidden_size"])
                mlflow.log_param("batch_size", config["batch_size"])
                mlflow.log_param("learning_rate", config["lr"])
                mlflow.log_param("epochs", config["epochs"])

                # Initialize model and loss as before
                model = MLP(input_dim=input_dim, hidden_size=config["hidden_size"], num_layers=num_layer)
                optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
                if loss_fn == "ce":
                    criterion = nn.BCEWithLogitsLoss()
                elif loss_fn == "wce":
                    weight = params[0]
                    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
                    mlflow.log_param("ce_weight", weight)
                elif loss_fn == "focal":
                    alpha, gamma = params
                    criterion = FocalLoss(alpha=alpha, gamma=gamma)
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("gamma", gamma)

                # Training loop
                best_val_f1 = 0.0
                best_model_state = None
                for epoch in range(config["epochs"]):
                    if (epoch+1)%10==0:
                        print(f"Epoch {epoch+1}/{config['epochs']} ...", end=" ", flush=True)
                    model.train()
                    total_loss = 0.0

                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        outputs = model(batch_x)
                        loss = criterion(outputs, batch_y.view(-1, 1))
                        loss.backward()
                        optimizer.step()
                        total_loss += loss.item()
                    epoch_loss = total_loss / len(train_loader)

                    # Validation
                    model.eval()
                    val_loss = 0.0
                    all_labels, all_preds = [], []
                    with torch.no_grad():
                        for batch_x, batch_y in val_loader:
                            outputs = model(batch_x)
                            loss = criterion(outputs, batch_y.view(-1, 1))
                            val_loss += loss.item()
                            probs = torch.sigmoid(outputs)
                            preds = (probs > 0.5).int().squeeze(1)
                            all_labels.extend(batch_y.numpy())
                            all_preds.extend(preds.numpy())
                    val_loss /= len(val_loader)
                    val_f1 = f1_score(all_labels, all_preds)

                    mlflow.log_metric("train_loss", epoch_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                    mlflow.log_metric("val_f1", val_f1, step=epoch)

                    if (epoch+1)%10==0:
                        print(f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
                    if val_f1 > best_val_f1:
                        best_val_f1 = val_f1
                        best_model_state = copy.deepcopy(model.state_dict())

                # Test evaluation
                if best_model_state:
                    model.load_state_dict(best_model_state)
                model.eval()
                all_labels, all_preds = [], []
                with torch.no_grad():
                    for batch_x, batch_y in test_loader:
                        outputs = model(batch_x)
                        probs = torch.sigmoid(outputs)
                        preds = (probs > 0.5).int().squeeze(1)
                        all_labels.extend(batch_y.numpy())
                        all_preds.extend(preds.numpy())

                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds)
                recall = recall_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds)
                cm = confusion_matrix(all_labels, all_preds)

                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)

                run_end_time = time.time() - run_start_time
                print("-"*90)
                print(f"RUN {current_run}/{total_runs} FINISHED in {run_end_time:.2f} seconds")
                print(f"Final Test Metrics -> Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                print("-"*90)

                mlflow.pytorch.log_model(model, name="MLP", input_example=X[:1])
    experiment_end_time = time.time() - experiment_start_time
    print(f"Experiment finished in {run_end_time:.2f} seconds")