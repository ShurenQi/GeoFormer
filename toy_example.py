from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import timm


MODEL_NAMES = ("gaformer_s18", "paformer_s18")
NUM_CLASSES = 4


class ToyPatternDataset(Dataset):
    _COLORS = torch.tensor(
        [
            [1.0, 0.1, 0.1],
            [0.1, 1.0, 0.1],
            [0.1, 0.1, 1.0],
            [0.8, 0.8, 0.1],
        ]
    )

    def __init__(self, sample_count: int, image_size: int, seed: int):
        if sample_count < NUM_CLASSES:
            raise ValueError(f"sample_count must be at least {NUM_CLASSES}")
        if image_size < 32 or image_size % 32 != 0:
            raise ValueError("image_size must be a multiple of 32 and at least 32")

        generator = torch.Generator().manual_seed(seed)
        images = 0.05 * torch.randn(
            sample_count, 3, image_size, image_size, generator=generator
        )
        labels = torch.arange(sample_count) % NUM_CLASSES
        patch_size = image_size // 2

        for index, label_tensor in enumerate(labels):
            label = int(label_tensor)
            row = (label // 2) * patch_size
            column = (label % 2) * patch_size
            color = self._COLORS[label].view(3, 1, 1)
            images[
                index, :, row : row + patch_size, column : column + patch_size
            ] += color

        self.images = images.clamp_(-1.0, 1.0)
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.images[index], self.labels[index]


@dataclass(frozen=True)
class ExperimentResult:
    model_name: str
    parameter_count: int
    initial_accuracy: float
    training_loss: float
    evaluation_loss: float
    final_accuracy: float
    training_seconds: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_model(model_name: str, device: torch.device) -> nn.Module:
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=NUM_CLASSES,
        drop_path_rate=0.0,
    )
    return model.to(device)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def train_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    final_loss = 0.0

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()

    for _ in range(epochs):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * images.shape[0]
            sample_count += images.shape[0]
        final_loss = loss_sum / sample_count

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    return final_loss, elapsed


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    loss_sum = 0.0
    correct = 0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss_sum += float(criterion(logits, labels))
        correct += int((logits.argmax(dim=1) == labels).sum())
        sample_count += images.shape[0]

    return loss_sum / sample_count, 100.0 * correct / sample_count


def run_experiment(
    model_name: str,
    train_dataset: Dataset,
    evaluation_dataset: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> ExperimentResult:
    seed_everything(args.seed)
    model = build_model(model_name, device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"[build] {model_name}: {parameter_count:,} parameters")

    initial_loader = make_loader(
        evaluation_dataset,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    _, initial_accuracy = evaluate_model(model, initial_loader, device)

    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        seed=args.seed,
    )
    training_loss, training_seconds = train_model(
        model,
        train_loader,
        device,
        args.epochs,
        args.learning_rate,
    )

    evaluation_loader = make_loader(
        evaluation_dataset,
        args.batch_size,
        shuffle=False,
        seed=args.seed,
    )
    evaluation_loss, final_accuracy = evaluate_model(model, evaluation_loader, device)
    print(
        f"[evaluate] {model_name}: loss={evaluation_loss:.4f}, "
        f"accuracy={final_accuracy:.1f}%"
    )

    return ExperimentResult(
        model_name=model_name,
        parameter_count=parameter_count,
        initial_accuracy=initial_accuracy,
        training_loss=training_loss,
        evaluation_loss=evaluation_loss,
        final_accuracy=final_accuracy,
        training_seconds=training_seconds,
    )


def print_comparison(results: list[ExperimentResult]) -> None:
    print("\nComparison (toy data; not paper accuracy)")
    print(
        f"{'model':<16} {'params':>12} {'initial':>10} {'train loss':>12} "
        f"{'eval loss':>11} {'final':>10} {'train s':>10}"
    )
    print("-" * 87)
    for result in results:
        print(
            f"{result.model_name:<16} "
            f"{result.parameter_count:>12,} "
            f"{result.initial_accuracy:>9.1f}% "
            f"{result.training_loss:>12.4f} "
            f"{result.evaluation_loss:>11.4f} "
            f"{result.final_accuracy:>9.1f}% "
            f"{result.training_seconds:>10.2f}"
        )

    best_accuracy = max(result.final_accuracy for result in results)
    best_names = ", ".join(
        result.model_name
        for result in results
        if result.final_accuracy == best_accuracy
    )
    smallest = min(results, key=lambda result: result.parameter_count)
    fastest = min(results, key=lambda result: result.training_seconds)
    print(
        f"\nHighest toy accuracy: {best_names} ({best_accuracy:.1f}%). "
        f"Fewest parameters: {smallest.model_name} "
        f"({smallest.parameter_count:,}). Fastest toy training: "
        f"{fastest.model_name} ({fastest.training_seconds:.2f}s)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Toy GAFormer/PAFormer build, train, and evaluation comparison"
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-samples", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for training normalization")
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    print(f"Device: {device}")

    train_dataset = ToyPatternDataset(args.train_samples, args.image_size, args.seed)
    evaluation_dataset = ToyPatternDataset(
        args.eval_samples, args.image_size, args.seed + 1
    )

    results = []
    for model_name in MODEL_NAMES:
        result = run_experiment(
            model_name,
            train_dataset,
            evaluation_dataset,
            args,
            device,
        )
        results.append(result)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_comparison(results)


if __name__ == "__main__":
    main()
