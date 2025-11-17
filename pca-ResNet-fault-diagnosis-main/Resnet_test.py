"""Utility script to evaluate a trained ResNet checkpoint on the test split."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from batch_loader import ClassDataset
from custom_tools import plot_confusion_matrix
from Resnet_model import ResNet


def evaluate_checkpoint(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint '{checkpoint_path}' does not exist.")

    test_set = ClassDataset(
        file=args.test_file,
        pca_file=args.pca_file,
        feature=args.feature_file,
        pca_add_feature=args.pca_add_feature_file,
        img_file=args.img_dir,
        sigma=args.sigma,
    )
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = ResNet(num_classes=args.num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    total_correct, total_samples = 0, 0
    all_targets, all_preds = [], []

    with torch.no_grad():
        for batch in test_loader:
            img_array = batch[1].to(device)
            target = batch[-1].to(device)

            output = model(img_array)
            preds = output.argmax(dim=1)

            total_correct += (preds == target).sum().item()
            total_samples += target.size(0)

            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    acc = total_correct / total_samples
    precision = precision_score(all_targets, all_preds, average="macro")
    recall = recall_score(all_targets, all_preds, average="macro")
    f1 = f1_score(all_targets, all_preds, average="macro")

    print(f"Accuracy: {acc:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1: {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(all_targets, all_preds, digits=4))

    if args.save_confusion:
        result_path = Path(args.save_confusion)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        plot_confusion_matrix(
            y_true=all_targets,
            y_pred=all_preds,
            savename=str(result_path),
            title=result_path.stem,
            classes=[str(i) for i in range(args.num_classes)],
        )
        print(f"Confusion matrix saved to {result_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ResNet checkpoint on the test split.")
    parser.add_argument("--checkpoint", default="result/ResNet.pt")
    parser.add_argument("--test_file", default="dataset/pickle_file/test.p")
    parser.add_argument("--pca_file", default="dataset/pickle_file/pca_train.p")
    parser.add_argument("--feature_file", default="dataset/pickle_file/features.p")
    parser.add_argument("--pca_add_feature_file", default="dataset/pickle_file/pca_add_feature.p")
    parser.add_argument("--img_dir", default="dataset/IMAGE")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--sigma", type=int, default=10)
    parser.add_argument(
        "--save_confusion",
        default="result/Confusion-Matrix-ResNet-test.png",
        help="Where to save the confusion matrix image. Set to '' to disable saving.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_checkpoint(parse_args())
