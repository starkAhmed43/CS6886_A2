"""Pull a wandb run's history and plot loss/accuracy curves for the Q1 report.

Fetches ``train/loss``, ``val/loss``, ``train/acc`` and ``val/acc`` from a
wandb run, collapses the NaN-interleaved history to one row per epoch, and
saves a two-panel PNG (loss on the left, accuracy on the right).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import wandb


def fetch_epoch_history(run_path: str) -> pd.DataFrame:
    api = wandb.Api()
    run = api.run(run_path)

    keys = ["epoch", "train/loss", "val/loss", "train/acc", "val/acc"]
    rows = list(run.scan_history(keys=keys))
    df = pd.DataFrame(rows)

    df = df.groupby("epoch").mean(numeric_only=True).reset_index()

    # Drop a trailing epoch row if it carries no metric values.
    metric_cols = ["train/loss", "val/loss", "train/acc", "val/acc"]
    if not df.empty and df.iloc[-1][metric_cols].isna().all():
        df = df.iloc[:-1]

    return df, run


def plot_curves(df: pd.DataFrame, run_name: str, out: str) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(df["epoch"], df["train/loss"], label="train/loss")
    ax_loss.plot(df["epoch"], df["val/loss"], label="val/loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True)

    ax_acc.plot(df["epoch"], df["train/acc"], label="train/acc")
    ax_acc.plot(df["epoch"], df["val/acc"], label="val/acc")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Top-1 Accuracy")
    ax_acc.legend()
    ax_acc.grid(True)

    fig.suptitle(f"Training curves: {run_name}")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        default="starkahmed43/cs6886-a2/qm6sg4fd",
        help="wandb run path as entity/project/run_id",
    )
    parser.add_argument(
        "--out",
        default="reports/q1_curves.png",
        help="output PNG path",
    )
    args = parser.parse_args()

    df, run = fetch_epoch_history(args.run)
    plot_curves(df, run.name, args.out)

    last = df.iloc[-1]
    print(f"Saved figure to {args.out}")
    print(
        f"Final epoch {int(last['epoch'])}: "
        f"train/loss={last['train/loss']:.4f}, val/loss={last['val/loss']:.4f}, "
        f"train/acc={last['train/acc']:.4f}, val/acc={last['val/acc']:.4f}"
    )


if __name__ == "__main__":
    main()
