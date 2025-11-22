import os
import sys

import matplotlib.pyplot as plt
from util import load_runs
from util import get_loss_label
from Visualization_Own.util import PLOT_DIR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def plot_final_f1():
    df, _ = load_runs()
    os.makedirs(PLOT_DIR, exist_ok=True)

    df["loss_label"] = df.apply(lambda r: get_loss_label(r)[0], axis=1)
    df["split"] = df["params.dataset_split"]
    df["f1"] = df["metrics.f1"]

    # Sort for consistent plotting
    df = df.sort_values(["loss_label", "split"])

    # group bar plot
    plt.figure(figsize=(12, 6))

    splits = df["split"].unique()
    x = range(len(df["loss_label"].unique()))
    width = 0.18

    color_map = {label: col for label, col in df[["loss_label", "loss_label"]].apply(
        lambda x: get_loss_label(df[df["loss_label"] == x[0]].iloc[0])[1], axis=1
    ).items()}

    for i, split in enumerate(splits):
        subset = df[df["split"] == split]
        f1_vals = subset["f1"].values

        plt.bar(
            [p + i*width for p in x],
            f1_vals,
            width=width,
            label=split,
        )

    plt.xticks([p + width for p in x], df["loss_label"].unique())
    plt.ylabel("Final Test F1")
    plt.title("Final Test F1 Across Data Splits")
    plt.legend()
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/final_f1.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    plot_final_f1()
