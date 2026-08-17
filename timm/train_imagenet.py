from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)

TRAINING_DEFAULTS: dict[str, Any] = {
    "data": None,
    "output": None,
    "experiment": "scratch_run01",
    "model": None,
    "num_classes": 1000,
    "batch_size": 256,
    "epochs": 300,
    "warmup_epochs": 5,
    "workers": 8,
    "opt": "adamw",
    "weight_decay": 0.05,
    "lr": 1.5e-3,
    "min_lr": 1e-6,
    "warmup_lr": 1e-6,
    "sched": "cosine",
    "mixup": 0.8,
    "cutmix": 1.0,
    "smoothing": 0.1,
    "drop_path": 0.1,
    "aa": "rand-m9-mstd0.5-inc1",
    "reprob": 0.25,
    "amp": True,
    "model_ema": True,
    "model_ema_decay": 0.99996,
    "log_interval": 50,
    "recovery_interval": 0,
    "input_size": (3, 224, 224),
}


def build_argv(config: dict[str, Any]) -> list[str]:
    argv: list[str] = []
    data_root = config.get("data")
    if data_root:
        argv.append(str(data_root))
    for name, value in config.items():
        if name == "data" or value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv.append(flag)
            argv.extend(str(item) for item in value)
        else:
            argv.extend((flag, str(value)))
    return argv


def has_flag(arguments: list[str], flag: str) -> bool:
    return flag in arguments or any(
        argument.startswith(flag + "=") for argument in arguments
    )


def find_latest_checkpoint(output_dir: str, experiment: str) -> str:
    checkpoint_dir = Path(output_dir) / experiment
    if not checkpoint_dir.exists():
        raise FileNotFoundError(checkpoint_dir)
    primary = checkpoint_dir / "checkpoint.pth.tar"
    if primary.is_file():
        return str(primary)
    pattern = re.compile(r"^checkpoint-(\d+)\.pth\.tar$")
    candidates = []
    for path in checkpoint_dir.iterdir():
        match = pattern.match(path.name)
        if path.is_file() and match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(checkpoint_dir)
    return str(max(candidates, key=lambda item: item[0])[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Train the included ImageNet models",
        allow_abbrev=False,
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment", default=TRAINING_DEFAULTS["experiment"])
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--drop-path", type=float)
    parser.add_argument("--reprob", type=float)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true")
    parser.add_argument("--print-only", action="store_true")
    args, remaining = parser.parse_known_args()
    args.remaining = remaining
    return args


def main() -> None:
    args = parse_args()
    config = dict(TRAINING_DEFAULTS)
    config.update(
        data=args.data,
        model=args.model,
        output=args.output,
        experiment=args.experiment,
    )
    for name in (
        "batch_size",
        "lr",
        "drop_path",
        "reprob",
        "workers",
        "epochs",
        "warmup_epochs",
        "weight_decay",
    ):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.no_amp:
        config["amp"] = False

    remaining = list(args.remaining)
    if not args.no_auto_resume and not has_flag(remaining, "--resume"):
        try:
            checkpoint = find_latest_checkpoint(args.output, args.experiment)
            remaining.extend(("--resume", checkpoint))
            print(f"[auto-resume] {checkpoint}")
        except FileNotFoundError:
            pass

    sys.argv = [sys.argv[0], *build_argv(config), *remaining]
    print("Running command:")
    print("  python " + " ".join(sys.argv))
    if args.print_only:
        return
    from timm.train import main as timm_train

    timm_train()


if __name__ == "__main__":
    main()
