import os
import sys

from util import load_runs
from Visualization_Own.util import PLOT_DIR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

def export_summary_csv():
    df, _ = load_runs()
    os.makedirs(PLOT_DIR, exist_ok=True)

    summary = df[[
        "run_id",
        "params.loss",
        "params.loss_param",
        "params.dataset_split",
        "metrics.f1",
        "metrics.accuracy",
        "metrics.precision",
        "metrics.recall",
        "metrics.roc_auc",
        "metrics.val_loss",
    ]]

    summary.to_csv(f"{PLOT_DIR}/summary_metrics.csv", index=False)
    print("Saved summary CSV to plots/summary_metrics.csv")


if __name__ == "__main__":
    export_summary_csv()
