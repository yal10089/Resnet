"""Training entry-point for the standalone ResNet fault-diagnosis baseline."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

from batch_loader import ClassDataset
from custom_tools import EarlyStopping, plot_confusion_matrix, plot_curve
from Resnet_model import ResNet


def set_seed(seed: int = 1) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0.0, 0

    for batch in loader:
        img_array = batch[1].to(device)
        target = batch[-1].to(device)

        output = model(img_array)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = target.size(0)
        total_loss += loss.item() * batch_size
        preds = output.argmax(dim=1)
        total_correct += (preds == target).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0.0, 0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for batch in loader:
            img_array = batch[1].to(device)
            target = batch[-1].to(device)

            output = model(img_array)
            loss = criterion(output, target)

            batch_size = target.size(0)
            total_loss += loss.item() * batch_size
            preds = output.argmax(dim=1)
            total_correct += (preds == target).sum().item()
            total_samples += batch_size

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = total_correct / total_samples
    metrics = {
        "loss": total_loss / total_samples,
        "acc": acc,
        "precision": precision_score(all_targets, all_preds, average="macro"),
        "recall": recall_score(all_targets, all_preds, average="macro"),
        "f1": f1_score(all_targets, all_preds, average="macro"),
        "targets": all_targets,
        "preds": all_preds,
    }
    return metrics


def train_resnet(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_set = ClassDataset(
        file=args.train_file,
        pca_file=args.pca_file,
        feature=args.feature_file,
        pca_add_feature=args.pca_add_feature_file,
        img_file=args.img_dir,
        sigma=args.sigma,
    )
    test_set = ClassDataset(
        file=args.test_file,
        pca_file=args.pca_file,
        feature=args.feature_file,
        pca_add_feature=args.pca_add_feature_file,
        img_file=args.img_dir,
        sigma=args.sigma,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = ResNet(num_classes=args.num_classes).to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    history = {"epoch": [], "train_loss": [], "train_acc": [], "test_acc": []}
    best_acc = 0.0

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / args.checkpoint_name

    start = time.time()
    for epoch in range(args.epochs):
        history["epoch"].append(epoch)
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(metrics["acc"])

        print(
            f"Epoch {epoch:03d} | train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
            f"test acc: {metrics['acc']:.4f}"
        )

        if metrics["acc"] > best_acc:
            best_acc = metrics["acc"]
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best accuracy: {best_acc:.4f}. Weights saved to {checkpoint_path}")

            plot_confusion_matrix(
                y_true=metrics["targets"],
                y_pred=metrics["preds"],
                savename=str(result_dir / f"Confusion-Matrix-{args.checkpoint_name}.png"),
                title=f"Confusion-Matrix-{args.checkpoint_name}",
                classes=[str(i) for i in range(args.num_classes)],
            )

            with open(result_dir / f"{args.checkpoint_name}.txt", "a", encoding="utf-8") as handle:
                handle.write(
                    "epoch:{},test_acc:{:.6f},precision:{:.6f},recall:{:.6f},f1:{:.6f}\n".format(
                        epoch, metrics["acc"], metrics["precision"], metrics["recall"], metrics["f1"]
                    )
                )

        early_stopping(metrics["acc"])
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    end = time.time()
    print(f"Training finished in {end - start:.2f}s")

    plot_curve(
        history["epoch"],
        history["train_loss"],
        history["train_acc"],
        history["test_acc"],
        savename=str(result_dir / f"train-loss-and-acc-{args.checkpoint_name}"),
        title=f"train-loss-and-acc-{args.checkpoint_name}",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the standalone ResNet baseline.")
    parser.add_argument("--train_file", default="dataset/pickle_file/train.p")
    parser.add_argument("--test_file", default="dataset/pickle_file/test.p")
    parser.add_argument("--pca_file", default="dataset/pickle_file/pca_train.p")
    parser.add_argument("--feature_file", default="dataset/pickle_file/features.p")
    parser.add_argument("--pca_add_feature_file", default="dataset/pickle_file/pca_add_feature.p")
    parser.add_argument("--img_dir", default="dataset/IMAGE")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--sigma", type=int, default=10)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--result_dir", default="result")
    parser.add_argument("--checkpoint_name", default="ResNet.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train_resnet(parse_args())
