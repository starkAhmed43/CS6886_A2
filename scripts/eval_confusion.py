"""Evaluate a trained CIFAR-10 checkpoint and report per-class accuracy + confusion matrix.

Used for the Q1 failure-mode analysis: which classes the model confuses most often.
"""

import os

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import argparse

import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.data.cifar10_datamodule import CIFAR10DataModule
from src.models.cifar_module import CIFARLitModule
from src.models.components.mobilenetv2 import mobilenetv2_cifar

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

DEFAULT_CKPT = "logs/train/runs/2026-09-07_10-08-10/checkpoints/epoch_193.ckpt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--out", type=str, default="reports/q1_confusion_matrix.png")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def load_model(ckpt: str, device: str) -> torch.nn.Module:
    """Load the model from a checkpoint, falling back to a manual state-dict load."""
    try:
        model = CIFARLitModule.load_from_checkpoint(ckpt, map_location=device)
    except Exception as exc:
        print(f"load_from_checkpoint failed ({exc!r}); falling back to manual net load.")
        net = mobilenetv2_cifar(num_classes=10, stride_relax=3)
        state_dict = torch.load(ckpt, map_location=device)["state_dict"]
        net_state_dict = {
            k[len("net.") :]: v for k, v in state_dict.items() if k.startswith("net.")
        }
        net.load_state_dict(net_state_dict)
        return net.to(device).eval()
    return model.to(device).eval()


@torch.no_grad()
def evaluate(model: torch.nn.Module, dataloader, device: str) -> torch.Tensor:
    """Run inference over the dataloader and return a 10x10 confusion matrix (rows=true)."""
    num_classes = len(CLASS_NAMES)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for images, targets in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = torch.argmax(logits, dim=1).cpu()
        indices = targets * num_classes + preds
        confusion += torch.bincount(indices, minlength=num_classes * num_classes).reshape(
            num_classes, num_classes
        )
    return confusion


def print_report(confusion: torch.Tensor) -> None:
    overall_acc = confusion.diag().sum().item() / confusion.sum().item()
    row_sums = confusion.sum(dim=1)
    per_class_acc = confusion.diag().float() / row_sums.float()

    print("\nPer-class accuracy:")
    print(f"{'class':<12}{'accuracy':>10}")
    for name, acc in zip(CLASS_NAMES, per_class_acc):
        print(f"{name:<12}{acc.item() * 100:>9.2f}%")
    print(f"\nOverall accuracy: {overall_acc * 100:.2f}%")

    off_diag = confusion.clone()
    off_diag.fill_diagonal_(0)
    flat = off_diag.flatten()
    top_k = torch.topk(flat, k=3)
    num_classes = len(CLASS_NAMES)
    print("\nTop-3 most-confused pairs (true -> pred: count):")
    for count, idx in zip(top_k.values, top_k.indices):
        true_idx, pred_idx = divmod(idx.item(), num_classes)
        print(f"{CLASS_NAMES[true_idx]} -> {CLASS_NAMES[pred_idx]}: {count.item()}")


def plot_confusion(confusion: torch.Tensor, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    matrix = confusion.numpy()
    num_classes = len(CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CIFAR-10 Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(num_classes):
        for j in range(num_classes):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    model = load_model(args.ckpt, args.device)

    datamodule = CIFAR10DataModule(
        data_dir=args.data_dir, batch_size=256, num_workers=8, pin_memory=False
    )
    datamodule.prepare_data()
    datamodule.setup()

    confusion = evaluate(model, datamodule.test_dataloader(), args.device)

    print_report(confusion)
    plot_confusion(confusion, args.out)
    print(f"\nSaved confusion matrix heatmap to {args.out}")


if __name__ == "__main__":
    main()
