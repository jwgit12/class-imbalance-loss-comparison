"""
plot_metrics.py
===============

This script reads MLflow runs from a given experiment and plots
training metrics (e.g., train_loss) for multiple loss functions
in a single figure. The plot is also saved as a PNG.
"""

import mlflow
import matplotlib.pyplot as plt
from utils import load_config


def plot_experiment_losses(config, loss_names, metric_name="train_loss", save_path="loss_plot.png"):
    """
    Pulls metrics from MLflow and plots them for multiple loss runs.

    Args:
        config (dict): Experiment configuration.
        loss_names (list): List of loss names to plot (e.g., ["ce", "wce", "focal"]).
        metric_name (str): Metric to plot. Default: "train_loss".
        save_path (str): File path to save the figure.
    """
    # Connect to MLflow server
    mlflow.set_tracking_uri(f"http://localhost:{config['mlflow_port']}")
    experiment = mlflow.get_experiment_by_name(config["mlflow_experiment_name"])

    if experiment is None:
        raise ValueError(f"Experiment '{config['mlflow_experiment_name']}' not found.")

    # Get all runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    plt.figure(figsize=(8, 6))

    for loss_type in loss_names:
        # Filter runs by name
        run = runs[runs['tags.mlflow.runName'] == f"{loss_type.upper()}_run"]
        if run.empty:
            print(f"No runs found for {loss_type}")
            continue
        run_id = run.iloc[0].run_id

        # Get metric history
        history = mlflow.tracking.MlflowClient().get_metric_history(run_id, metric_name)
        steps = [h.step for h in history]
        values = [h.value for h in history]

        plt.scatter(steps, values, label=loss_type.upper())

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(metric_name.replace("_", " ").title(), fontsize=12)
    plt.title(f"{metric_name.replace('_', ' ').title()} Comparison", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")
    plt.show()


if __name__ == "__main__":
    """
    Run with `python plot_metrics.py`.
    """
    config = load_config()

    plot_experiment_losses(
        config,
        loss_names=["ce", "wce", "focal"],
        metric_name="train_loss",
        save_path="loss_comparison.png"
    )
