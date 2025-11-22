import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from util import load_runs
from Visualization_Own.util import PLOT_DIR

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

os.makedirs(PLOT_DIR, exist_ok=True)
df, client = load_runs()

def get_best_runs_per_config(all_runs: pd.DataFrame) -> pd.DataFrame:
    df = all_runs.copy()

    df['split'] = df['params.dataset_split']
    df['imbalance_ratio'] = df['params.imbalance_ratio'].astype(float)
    df['loss_function'] = df['params.loss']
    df['loss_param'] = df['params.loss_param'].astype(float)

    # Group by dataset + imbalance + loss (focal_x.x included!)
    group_cols = ['split', 'imbalance_ratio', 'loss_function', 'loss_param']

    best_idx = df.groupby(group_cols)['metrics.val_f1'].idxmax()
    best = df.loc[best_idx].reset_index(drop=True)

    return best


def fetch_val_f1_history(best_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in best_df.iterrows():
        run_id = row['run_id']
        hist = client.get_metric_history(run_id, "val_f1")

        for m in hist:
            rows.append({
                "step": m.step,
                "val_f1": m.value,
                "split": row["split"],
                "imbalance_ratio": float(row['imbalance_ratio']),
                "loss_function": row['loss_function'],
                "loss_param": float(row["loss_param"]),
            })

    return pd.DataFrame(rows)


def make_loss_label(loss, gamma):
    if loss == "focal":
        return f"focal (γ={gamma:.1f})"
    return loss


def plot_best_convergence(best_hist: pd.DataFrame, title: str):
    if best_hist.empty:
        print("No data to plot.")
        return

    df = best_hist.copy()
    df["loss_display"] = df.apply(
        lambda r: make_loss_label(r["loss_function"], r["loss_param"]),
        axis=1
    )

    # Sort splits by imbalance ratio
    split_order = df.groupby("split")["imbalance_ratio"].first().sort_values().index.tolist()

    # Column = dataset split
    g = sns.relplot(
        data=df,
        x="step",
        y="val_f1",
        hue="loss_display",
        kind="line",
        col="split",
        col_order=split_order,
        col_wrap=len(split_order),
        facet_kws={"sharey": True, "sharex": True},
        height=3.5,
    )

    # Automatic layout adjustment
    g.fig.set_constrained_layout(True)
    g.fig.suptitle(title, fontsize=14, y=1.15)

    # Titles & best loss below each subplot
    for ax, split in zip(g.axes.flatten(), split_order):
        split_df = df[df["split"] == split]
        imbalance = split_df["imbalance_ratio"].iloc[0]
        ax.set_title(f"Split: {split}\nImbalance = {imbalance:.2f}")

        # Collect all loss names with parameters for this split
        loss_labels = split_df["loss_display"].unique()
        losses_text = " | ".join(loss_labels)

        # Display below the plot
        ax.text(
            0.5, -0.25,  # relative coordinates: x=center, y=below
            f"Losses used: {losses_text}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9
        )

    # Global labels
    g.set_axis_labels("Epoch", "Validation F1")

    # Remove old legend
    g._legend.remove()
    axes = g.axes.flatten() if hasattr(g.axes, "flatten") else [g.axes]
    handles, labels = axes[0].get_legend_handles_labels()

    # Horizontal legend above plots
    g.figure.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=len(labels),
        fontsize=9,
        title_fontsize=10,
        bbox_to_anchor=(0.5, 1.02)
    )

    # Final safety pass
    g.figure.canvas.draw()
    plt.tight_layout()

    # Save figure
    save_path = f"{PLOT_DIR}/f1_convergence_best.png"
    g.fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")


if __name__ == "__main__":
    #plot_f1_convergence_subplots()
    best_runs = get_best_runs_per_config(df)
    best_hist = fetch_val_f1_history(best_runs)

    plot_best_convergence(best_hist, "Validation F1 Convergence for Loss Functions")
